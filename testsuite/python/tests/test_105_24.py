############################################################################
# Copyright (C) SchedMD LLC.
############################################################################
"""Test an external --no-kill job is killed, never shrunk, when it loses a node.

A --no-kill job that loses a node usually keeps running on its remaining
nodes, and the lost node's resources are released. slurmctld cannot know
that an external job's workload on that node stopped, so it must not release
them: an external job either keeps its whole allocation or is killed.

A multi-node external --no-kill job keeps its whole allocation when one of
its nodes goes DOWN. Setting that DOWN node FUTURE (allowed from DOWN, not
from ALLOCATED) does not keep allocations, so the job is killed there. A
batch --no-kill job losing a DOWN node keeps running on its remaining node,
as before. A hetjob does not keep its allocation through a DOWN node either:
a node failure fails every component of it together.

The nodes are dynamic, so a node left FUTURE is simply deleted afterwards.
"""

import pytest

import atf

slurm_user = atf.properties["slurm-user"]

node_cpus = 4
job_cpus = 2
dyn_nodes = {"dyn1": 61710, "dyn2": 61711}


@pytest.fixture(scope="module", autouse=True)
def setup():
    atf.require_version(
        (26, 11),
        "sbin/slurmctld",
        reason="External jobs are never shrunk by a node failure in 26.11",
    )

    atf.require_config_parameter("SelectType", "select/cons_tres")
    atf.require_config_parameter("SelectTypeParameters", "CR_CPU")
    atf.require_config_parameter("MaxNodeCount", 8)
    atf.require_config_parameter(
        "PartitionName", {"primary": {"Nodes": "ALL", "Default": "YES"}}
    )
    atf.require_config_parameter_includes("SchedulerParameters", ("sched_interval", 1))
    # Only the backfill scheduler starts hetjobs
    atf.require_config_parameter_includes("SchedulerParameters", ("bf_interval", 1))
    atf.require_config_parameter("JobRequeue", 1)

    # This module starts no slurmdbd, so finished jobs are checked through
    # scontrol
    atf.require_accounting(enabled=False)
    atf.require_config_parameter("MinJobAge", 300, ">=")

    atf.require_slurm_running()


@pytest.fixture(scope="function")
def nodes():
    """Yield two IDLE dynamic nodes, deleted afterwards."""

    for name, port in dyn_nodes.items():
        atf.start_dynamic_slurmd(name, extra_conf=f"Port={port} CPUs={node_cpus}")

    yield list(dyn_nodes)

    atf.cancel_all_jobs(quiet=True)
    for name in dyn_nodes:
        atf.stop_slurmd(name, quiet=True)
        atf.delete_node(name, quiet=True)


def get_node(node):
    return atf.get_nodes(quiet=True)[node]


def set_node_state(node, state):
    atf.run_command(
        f"scontrol update nodename={node} state={state} reason=test_105_24",
        user=slurm_user,
        fatal=True,
    )
    atf.wait_for_node_state(node, state.upper(), fatal=True)


def test_external_killed_not_shrunk(nodes):
    """A multi-node external --no-kill job is killed when it loses a node."""

    job_id = atf.submit_job_sbatch(
        f"--external --no-kill -N2 -n{2 * job_cpus} --ntasks-per-node={job_cpus}"
        f" -w {','.join(nodes)} -t10",
        fatal=True,
    )
    atf.wait_for_job_state(job_id, "RUNNING", fatal=True)

    # DOWN keeps the whole allocation...
    set_node_state(nodes[1], "down")
    job = atf.get_job(job_id)
    assert job["JobState"] == "RUNNING", f"Job {job_id} did not keep running: {job}"
    assert job["NumNodes"] == 2, f"Job {job_id} was shrunk: {job}"

    # ...but FUTURE does not, and the job must not keep running without it
    set_node_state(nodes[1], "future")

    assert atf.wait_for_job_state(
        job_id, "NODE_FAIL"
    ), f"Job {job_id} should be killed, not keep running without {nodes[1]}"
    job = atf.get_job(job_id)
    assert job["NumNodes"] == 2, f"Job {job_id} was shrunk: {job}"
    assert atf.repeat_until(
        lambda: get_node(nodes[0])["alloc_cpus"], lambda cpus: cpus == 0
    ), f"{nodes[0]} still has CPUs allocated after the job was killed"


def test_batch_shrunk(nodes):
    """A multi-node batch --no-kill job losing a node keeps running."""

    job_id = atf.submit_job_sbatch(
        f"--no-kill -N2 -n{2 * job_cpus} --ntasks-per-node={job_cpus}"
        f" -w {','.join(nodes)} -t10 --wrap 'sleep 300'",
        fatal=True,
    )
    atf.wait_for_job_state(job_id, "RUNNING", fatal=True)
    batch_host = atf.get_job_parameter(job_id, "BatchHost")
    lost_node = next(n for n in nodes if n != batch_host)

    set_node_state(lost_node, "down")

    job = atf.get_job(job_id)
    assert job["JobState"] == "RUNNING", f"Job {job_id} did not keep running: {job}"
    assert (
        job["NodeList"] == batch_host
    ), f"Job {job_id} should only keep {batch_host}: {job['NodeList']}"


def test_external_hetjob_fails(nodes):
    """An external --no-kill hetjob fails as a whole when a node goes DOWN."""

    leader_id = atf.submit_job_sbatch(
        f"--external --no-kill -N1 -n{job_cpus} -w {nodes[0]} -t10"
        f" : --external --no-kill -N1 -n{job_cpus} -w {nodes[1]} -t10",
        fatal=True,
    )
    component_ids = [leader_id, leader_id + 1]
    for job_id in component_ids:
        atf.wait_for_job_state(job_id, "RUNNING", fatal=True)

    set_node_state(nodes[1], "down")

    for job_id in component_ids:
        assert atf.wait_for_job_state(
            job_id, "NODE_FAIL"
        ), f"Hetjob component {job_id} should fail with {nodes[1]}"
    for node in nodes:
        assert atf.repeat_until(
            lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == 0
        ), f"{node} still has CPUs allocated after the hetjob failed"
