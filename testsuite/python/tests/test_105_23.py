############################################################################
# Copyright (C) SchedMD LLC.
############################################################################
"""Test external --no-kill jobs keep their allocation through a DOWN node.

An external job (sbatch --external) runs outside of slurmd, so its node going
DOWN says nothing about whether its workload still runs. When it was
submitted with --no-kill it keeps running with its whole allocation,
including its CPUs on the DOWN node, until it is cancelled or requeued. The
node takes no new work while it is DOWN, and returns to service ALLOCATED.

Clients report a node with some CPUs allocated as MIXED whatever its state
in slurmctld, so the tests that check ALLOCATED versus IDLE use a job taking
the whole node, and the others check the node's CPUAlloc and which jobs can
still start on it.
"""

import pytest

import atf

slurm_user = atf.properties["slurm-user"]

node_cpus = 4
job_cpus = 2

# Short enough for the "Not responding" DOWN to come quickly
slurmd_timeout = 5

dyn_node = "dyn1"
dyn_port = 61700


@pytest.fixture(scope="module", autouse=True)
def setup():
    atf.require_version(
        (26, 11),
        "sbin/slurmctld",
        reason="External --no-kill jobs keep their allocation through a DOWN node in 26.11",
    )

    atf.require_nodes(2, [("CPUs", node_cpus)])
    atf.require_config_parameter("SelectType", "select/cons_tres")
    atf.require_config_parameter("SelectTypeParameters", "CR_CPU")
    atf.require_config_parameter("SlurmdTimeout", slurmd_timeout)
    # A node DOWN for "Not responding" returns to service when slurmd
    # registers again
    atf.require_config_parameter("ReturnToService", 1)
    atf.require_config_parameter("JobRequeue", 1)
    # Lets test_reboot_waits ask for a reboot; no node is actually rebooted
    atf.require_config_parameter("RebootProgram", "/bin/true")

    # Room for the dynamic node of test_delete_refused, which joins the
    # default partition through Nodes=ALL
    atf.require_config_parameter("MaxNodeCount", 8)
    atf.require_config_parameter(
        "PartitionName", {"primary": {"Nodes": "ALL", "Default": "YES"}}
    )
    atf.require_config_parameter_includes("SchedulerParameters", ("sched_interval", 1))
    # test_restart_with_reboot_pending needs backfill to run within its
    # window: only backfill would start a job on a reboot-pending node
    atf.require_config_parameter_includes("SchedulerParameters", ("bf_interval", 1))

    # This module starts no slurmdbd, so finished jobs are checked through
    # scontrol
    atf.require_accounting(enabled=False)
    atf.require_config_parameter("MinJobAge", 300, ">=")

    atf.require_slurm_running()


@pytest.fixture(scope="function")
def node():
    """Yield an IDLE node, and leave it IDLE with its slurmd running."""

    node = atf.require_nodes(2, [("CPUs", node_cpus)])[1]
    atf.wait_for_node_state(node, "IDLE", fatal=True)

    yield node

    atf.cancel_all_jobs(quiet=True)
    if not atf.slurmd_pids(node):
        atf.start_slurmd(node, quiet=True)
    atf.run_command(
        f"scontrol update nodename={node} state=resume",
        user=slurm_user,
        quiet=True,
    )
    atf.wait_for_node_state(node, "IDLE", fatal=True)


def submit_external(node, cpus, args=""):
    """Submit an external job using cpus of node and wait for it to run."""

    job_id = atf.submit_job_sbatch(
        f"--external -N1 -n{cpus} -w {node} -t10 {args}", fatal=True
    )
    atf.wait_for_job_state(job_id, "RUNNING", fatal=True)
    return job_id


def get_node(node):
    return atf.get_nodes(quiet=True)[node]


def resume_node(node):
    atf.run_command(
        f"scontrol update nodename={node} state=resume", user=slurm_user, fatal=True
    )


def set_node_down(node, how="scontrol"):
    """Set node DOWN explicitly, or by stopping its slurmd past SlurmdTimeout."""

    if how == "scontrol":
        atf.run_command(
            f"scontrol update nodename={node} state=down reason=test_105_23",
            user=slurm_user,
            fatal=True,
        )
        atf.wait_for_node_state(node, "DOWN", fatal=True)
    else:
        atf.stop_slurmd(node)
        atf.wait_for_node_state(node, "DOWN", timeout=slurmd_timeout * 6, fatal=True)
        reason = get_node(node)["reason"]
        assert (
            reason == "Not responding"
        ), f"{node} went DOWN for {reason!r}, not 'Not responding'"


def assert_job_retained(job_id, node, job_before, cpus, state="RUNNING"):
    """Assert the job runs as before and node still has its CPUs allocated."""

    job = atf.get_job(job_id)
    assert job["JobState"] == state, f"Job {job_id} did not stay {state}: {job}"
    for key in ("NodeList", "NumNodes", "NumCPUs", "AllocTRES"):
        assert (
            job[key] == job_before[key]
        ), f"Job {job_id} {key} changed from {job_before[key]} to {job[key]}"
    assert job["Restarts"] == 0, f"Job {job_id} was requeued: {job}"

    alloc_cpus = get_node(node)["alloc_cpus"]
    assert (
        alloc_cpus == cpus
    ), f"{node} should still have {cpus} CPUs allocated, not {alloc_cpus}"


def submit_probe(node, cpus, user=None):
    return atf.submit_job_sbatch(
        f"-N1 -n{cpus} -w {node} -t1 -o /dev/null --wrap 'sleep 30'",
        user=user,
        fatal=True,
    )


def probe_starts(node, cpus, user=None):
    """Return whether a job asking for cpus of node starts running."""

    probe_id = submit_probe(node, cpus, user)
    started = atf.wait_for_job_state(probe_id, "RUNNING", timeout=30, quiet=True)
    atf.cancel_jobs([probe_id], user=user, fatal=True)
    return started


def probe_blocked(node, cpus, user=None):
    """Return why a job asking for cpus of node stays pending.

    Waits until the scheduler has considered the job, that is until it has
    a pending reason, and returns that reason, or None if the job started.
    """

    probe_id = submit_probe(node, cpus, user)
    atf.repeat_until(
        lambda: atf.get_job(probe_id, quiet=True),
        lambda job: job["JobState"] == "RUNNING" or job.get("Reason", "None") != "None",
        timeout=30,
        fatal=True,
    )
    job = atf.get_job(probe_id, quiet=True)
    atf.cancel_jobs([probe_id], user=user, fatal=True)
    if job["JobState"] == "RUNNING":
        return None
    return job["Reason"]


@pytest.mark.parametrize("how", ["scontrol", "slurmd_timeout"])
def test_retained_through_down(node, how):
    """An external --no-kill job keeps the node, which comes back ALLOCATED."""

    job_id = submit_external(node, node_cpus, "--no-kill")
    job_before = atf.get_job(job_id)

    set_node_down(node, how)
    assert_job_retained(job_id, node, job_before, node_cpus)
    assert probe_blocked(node, 1), f"A job started on DOWN {node}"

    # Back in service, the node is ALLOCATED to the job, not IDLE
    if how == "scontrol":
        resume_node(node)
    else:
        atf.start_slurmd(node)
    atf.wait_for_node_state(node, "ALLOCATED", fatal=True)
    state = get_node(node)["state"]
    assert "IDLE" not in state and "DOWN" not in state, f"{node} is {state}"
    assert_job_retained(job_id, node, job_before, node_cpus)
    assert probe_blocked(node, 1), f"A job started on {node}, allocated to job {job_id}"

    # Ending the job frees the node
    atf.cancel_jobs([job_id], fatal=True)
    atf.wait_for_node_state(node, "IDLE", fatal=True)
    assert get_node(node)["alloc_cpus"] == 0


def test_partial_allocation(node):
    """A DOWN node holding part of its CPUs reports DOWN, and keeps them."""

    job_id = submit_external(node, job_cpus, "--no-kill")
    job_before = atf.get_job(job_id)

    set_node_down(node)
    assert_job_retained(job_id, node, job_before, job_cpus)

    # Not a state inferred from its allocated CPUs
    state = get_node(node)["state"]
    assert "DOWN" in state, f"{node} is not DOWN: {state}"
    assert "MIXED" not in state, f"DOWN {node} reported as MIXED: {state}"
    assert probe_blocked(node, 1), f"A job started on DOWN {node}"

    # Back in service, its free CPUs can be used but not the retained ones
    resume_node(node)
    atf.wait_for_node_state(node, "MIXED", fatal=True)
    assert_job_retained(job_id, node, job_before, job_cpus)
    assert probe_blocked(
        node, node_cpus
    ), f"A job got all CPUs of {node}, {job_cpus} of which job {job_id} holds"
    assert probe_starts(
        node, node_cpus - job_cpus
    ), f"A job could not use the free CPUs of {node}"


@pytest.mark.parametrize("command", ["scancel", "requeuehold"])
def test_release_on_down_node(node, command):
    """Ending a retained job on a DOWN node leaves nothing allocated on it."""

    job_id = submit_external(node, job_cpus, "--no-kill --requeue")
    job_before = atf.get_job(job_id)
    set_node_down(node)
    assert_job_retained(job_id, node, job_before, job_cpus)

    if command == "scancel":
        atf.cancel_jobs([job_id], fatal=True)
    else:
        atf.run_command(f"scontrol requeuehold {job_id}", user=slurm_user, fatal=True)
        atf.wait_for_job_state(job_id, "PENDING", fatal=True)

    assert atf.repeat_until(
        lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == 0
    ), f"{node} still has CPUs allocated after the job was released"
    assert "DOWN" in get_node(node)["state"], f"{node} did not stay DOWN"

    resume_node(node)
    atf.wait_for_node_state(node, "IDLE", fatal=True)
    assert probe_starts(node, node_cpus), f"A job could not get all CPUs of {node}"


def test_owner_cleared_on_down_node(node):
    """A retained --exclusive=user job ending on a DOWN node frees the node."""

    job_id = submit_external(node, job_cpus, "--no-kill --exclusive=user")
    job_before = atf.get_job(job_id)

    # The node stays reserved to the job's user through the DOWN
    set_node_down(node)
    resume_node(node)
    atf.wait_for_node_state(node, "MIXED", fatal=True)
    assert_job_retained(job_id, node, job_before, job_cpus)
    assert probe_blocked(
        node, 1, user=slurm_user
    ), f"Another user's job started on {node}, reserved to job {job_id}'s user"

    # Once the job ends while the node is DOWN, nobody owns the node. A
    # later slurmd registration of an IDLE node would also clear it, so
    # check while the node is still DOWN.
    set_node_down(node)
    owner = get_node(node)["owner"]
    assert (
        owner == atf.get_user_name()
    ), f"DOWN {node} should still be owned by job {job_id}'s user, not {owner!r}"
    atf.cancel_jobs([job_id], fatal=True)
    assert atf.repeat_until(
        lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == 0
    ), f"{node} still has CPUs allocated after the job was cancelled"
    owner = get_node(node)["owner"]
    assert not owner, f"{node} is still owned by {owner!r} after job {job_id} ended"
    resume_node(node)
    atf.wait_for_node_state(node, "IDLE", fatal=True)
    assert probe_starts(
        node, node_cpus, user=slurm_user
    ), f"Another user's job could not use {node} after job {job_id} ended"


def test_owner_released_per_job(node):
    """Each retained --exclusive=user job ending releases its share of the node.

    A job of the same user that does not ask for the node exclusively keeps
    the node busy, so its ownership is not just cleared when it goes IDLE.
    """

    log_file = atf.get_config_parameter("SlurmctldLogFile", live=False, quiet=True)
    underflow = f"grep -c 'owner_job_cnt underflow' {log_file} || true"
    underflows_before = int(atf.run_command_output(underflow, user=slurm_user))
    user = atf.get_user_name()

    exclusive_ids = [
        submit_external(node, 1, "--no-kill --exclusive=user") for _ in range(2)
    ]
    shared_id = submit_external(node, 1, "--no-kill")
    set_node_down(node)

    # One exclusive job ending while the node is DOWN leaves it owned by the
    # other, the last one ending releases it
    atf.cancel_jobs([exclusive_ids[0]], fatal=True)
    assert atf.repeat_until(
        lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == 2
    ), f"{node} should have 2 CPUs allocated after job {exclusive_ids[0]} ended"
    owner = get_node(node)["owner"]
    assert (
        owner == user
    ), f"{node} should still be owned by job {exclusive_ids[1]}'s user, not {owner!r}"

    atf.cancel_jobs([exclusive_ids[1]], fatal=True)
    assert atf.repeat_until(
        lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == 1
    ), f"{node} should have 1 CPU allocated to job {shared_id}"
    owner = get_node(node)["owner"]
    assert (
        not owner
    ), f"{node} is still owned by {owner!r} after its exclusive jobs ended"

    # Back in service, another user's job can use its free CPUs next to the
    # job that still runs there
    resume_node(node)
    atf.wait_for_node_state(node, "MIXED", fatal=True)
    assert probe_starts(
        node, 1, user=slurm_user
    ), f"Another user's job could not use the free CPUs of {node}"

    atf.cancel_jobs([shared_id], fatal=True)
    atf.wait_for_node_state(node, "IDLE", fatal=True)
    underflows = int(atf.run_command_output(underflow, user=slurm_user))
    assert underflows == underflows_before, "slurmctld logged owner_job_cnt underflow"


def test_suspend_while_down(node):
    """A suspended job holding a DOWN node resumes only once the node is up.

    Resuming it while the node is DOWN fails without allocating anything
    again, so the job stays suspended with its allocation as it was.
    """

    job_id = submit_external(node, job_cpus, "--no-kill")
    job_before = atf.get_job(job_id)
    set_node_down(node)
    assert_job_retained(job_id, node, job_before, job_cpus)

    atf.run_command(f"scontrol suspend {job_id}", user=slurm_user, fatal=True)
    atf.wait_for_job_state(job_id, "SUSPENDED", fatal=True)
    assert "DOWN" in get_node(node)["state"], f"{node} did not stay DOWN"
    suspended_cpus = get_node(node)["alloc_cpus"]

    result = atf.run_command(f"scontrol resume {job_id}", user=slurm_user, xfail=True)
    assert (
        "Required node not available" in result["stderr"]
    ), f"Resuming job {job_id} on DOWN {node} did not fail as expected: {result}"
    assert (
        atf.get_job_parameter(job_id, "JobState") == "SUSPENDED"
    ), f"Job {job_id} left SUSPENDED although {node} is DOWN"
    alloc_cpus = get_node(node)["alloc_cpus"]
    assert alloc_cpus == suspended_cpus, (
        f"The failed resume changed the CPUs allocated on {node} from "
        f"{suspended_cpus} to {alloc_cpus}"
    )

    # Once the node is up, the job resumes with its allocation
    resume_node(node)
    atf.wait_for_node_state(node, "IDLE", fatal=True)
    atf.run_command(f"scontrol resume {job_id}", user=slurm_user, fatal=True)
    atf.wait_for_job_state(job_id, "RUNNING", fatal=True)
    job = atf.get_job(job_id)
    for key in ("NodeList", "NumNodes", "NumCPUs", "AllocTRES"):
        assert (
            job[key] == job_before[key]
        ), f"Job {job_id} {key} changed from {job_before[key]} to {job[key]}"
    assert atf.repeat_until(
        lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == job_cpus
    ), f"{node} should have {job_cpus} CPUs allocated to resumed job {job_id}"


def test_reboot_waits(node):
    """A DOWN node is rebooted only once its retained allocation is released."""

    job_id = submit_external(node, node_cpus, "--no-kill")
    job_before = atf.get_job(job_id)
    set_node_down(node)

    atf.run_command(f"scontrol reboot {node}", user=slurm_user, fatal=True)
    try:
        # The reboot is issued at once on an empty DOWN node; give it a few
        # passes of the power save thread
        assert not atf.repeat_until(
            lambda: get_node(node)["state"],
            lambda state: "REBOOT_ISSUED" in state,
            timeout=10,
        ), f"{node} was rebooted while job {job_id} holds it"
        state = get_node(node)["state"]
        assert "REBOOT_REQUESTED" in state, f"{node} lost the reboot request: {state}"
        assert_job_retained(job_id, node, job_before, node_cpus)

        # Once the job ends, the reboot goes ahead
        atf.cancel_jobs([job_id], fatal=True)
        assert atf.repeat_until(
            lambda: get_node(node)["state"],
            lambda state: "REBOOT_ISSUED" in state,
            timeout=30,
        ), f"{node} was not rebooted after job {job_id} ended"
    finally:
        atf.run_command(f"scontrol cancel_reboot {node}", user=slurm_user, quiet=True)


@pytest.mark.parametrize("state", ["RUNNING", "SUSPENDED"])
def test_restart_with_reboot_pending(node, state):
    """After a restart, a DOWN node waiting to reboot stays out of service.

    A reboot with nextstate=resume waits for the retained job to end, whether
    the job runs or was suspended after the node went DOWN. A restarted
    slurmctld must not put the node back in service meanwhile.
    """

    job_id = submit_external(node, job_cpus, "--no-kill")
    job_before = atf.get_job(job_id)
    set_node_down(node)
    if state == "SUSPENDED":
        atf.run_command(f"scontrol suspend {job_id}", user=slurm_user, fatal=True)
        atf.wait_for_job_state(job_id, "SUSPENDED", fatal=True)
        cpus = get_node(node)["alloc_cpus"]
    else:
        cpus = job_cpus
    atf.run_command(
        f"scontrol reboot nextstate=resume {node}", user=slurm_user, fatal=True
    )
    # Queued before the restart, so the scheduler considers it as soon as it
    # starts, before the node answers its first ping
    probe_id = submit_probe(node, 1)
    try:
        atf.restart_slurmctld()
        atf.wait_for_job_state(job_id, state, fatal=True)
        assert not atf.wait_for_job_state(
            probe_id, "RUNNING", timeout=15, quiet=True
        ), f"Job {probe_id} started on DOWN {node} after the restart"
        node_state = get_node(node)["state"]
        assert (
            "DOWN" in node_state and "REBOOT_REQUESTED" in node_state
        ), f"{node} should stay DOWN with its reboot pending: {node_state}"
        assert_job_retained(job_id, node, job_before, cpus, state)
    finally:
        atf.run_command(f"scontrol cancel_reboot {node}", user=slurm_user, quiet=True)
        atf.cancel_jobs([probe_id], quiet=True)


def test_restart_keeps_retained(node):
    """A retained allocation on a DOWN node survives a slurmctld restart."""

    job_id = submit_external(node, node_cpus, "--no-kill")
    job_before = atf.get_job(job_id)
    set_node_down(node)

    atf.restart_slurmctld()

    atf.wait_for_job_state(job_id, "RUNNING", fatal=True)
    assert_job_retained(job_id, node, job_before, node_cpus)
    assert "DOWN" in get_node(node)["state"], f"{node} did not stay DOWN"
    assert probe_blocked(node, 1), f"A job started on DOWN {node}"

    resume_node(node)
    atf.wait_for_node_state(node, "ALLOCATED", fatal=True)
    assert_job_retained(job_id, node, job_before, node_cpus)


@pytest.mark.parametrize(
    "args",
    [
        "--external",
        "--no-kill --no-requeue --wrap 'sleep 300'",
    ],
    ids=["external_without_no_kill", "batch_with_no_kill"],
)
def test_other_jobs_fail(node, args):
    """Jobs other than external --no-kill ones still fail with their node."""

    job_id = atf.submit_job_sbatch(
        f"-N1 -n{job_cpus} -w {node} -t10 {args}", fatal=True
    )
    atf.wait_for_job_state(job_id, "RUNNING", fatal=True)

    set_node_down(node)

    assert atf.wait_for_job_state(
        job_id, "NODE_FAIL"
    ), f"Job {job_id} should end NODE_FAIL when its node goes DOWN"
    assert atf.repeat_until(
        lambda: get_node(node)["alloc_cpus"], lambda cpus: cpus == 0
    ), f"{node} still has CPUs allocated after its job failed"


@pytest.fixture(scope="function")
def dynamic_node():
    atf.start_dynamic_slurmd(dyn_node, extra_conf=f"Port={dyn_port} CPUs={node_cpus}")

    yield dyn_node

    atf.cancel_all_jobs(quiet=True)
    atf.stop_slurmd(dyn_node, quiet=True)
    atf.delete_node(dyn_node, quiet=True)


def test_delete_refused(dynamic_node):
    """A DOWN node holding a retained allocation cannot be deleted."""

    job_id = submit_external(dynamic_node, job_cpus, "--no-kill")
    job_before = atf.get_job(job_id)
    set_node_down(dynamic_node)

    result = atf.run_command(
        f"scontrol delete nodename={dynamic_node}", user=slurm_user, xfail=True
    )
    assert result["exit_code"] != 0, f"{dynamic_node} was deleted while holding a job"
    assert dynamic_node in atf.get_nodes(quiet=True), f"{dynamic_node} is gone"
    assert_job_retained(job_id, dynamic_node, job_before, job_cpus)

    # Once the job ends, the node can be deleted
    atf.cancel_jobs([job_id], fatal=True)
    atf.stop_slurmd(dynamic_node)
    result = atf.run_command(
        f"scontrol delete nodename={dynamic_node}", user=slurm_user
    )
    assert result["exit_code"] == 0, result["stderr"]
