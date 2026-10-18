import warnings


def test_reference_patches_are_present_in_threads(
    reference_patches,
    pipeline_results,
):
    message_ids = pipeline_results["message_ids"]

    missing = [
        patch["message_id"]
        for patch in reference_patches
        if "message_id" in patch and patch["message_id"] not in message_ids
    ]

    if missing:
        warnings.warn(UserWarning(
            f"{len(missing)} reference patches were not found "
            f"in any output thread:\n" + "\n".join(missing),
        ))


def test_reference_commits_are_found(
    reference_patches,
    pipeline_results,
):
    commit_hashes = pipeline_results["commit_hashes"]

    expected_hashes = {
        patch["commit_hash"]
        for patch in reference_patches
        if patch.get("commit_hash") is not None
    }

    missing = expected_hashes - commit_hashes

    if missing:
        warnings.warn(UserWarning(
            f"{len(missing)} reference commits were not found:\n"
            + "\n".join(sorted(missing)),
        ))
