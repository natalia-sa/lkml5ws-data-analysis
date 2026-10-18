import warnings


def test_message_commit_associations(
    reference_patches,
    pipeline_results,
):
    commit_by_message = pipeline_results["commit_by_message"]

    incorrect = []

    for patch in reference_patches:
        message_id = patch.get("message_id")
        expected_hash = patch.get("commit_hash")

        if not message_id or not expected_hash:
            continue

        actual_hashes = commit_by_message.get(message_id, set())

        if expected_hash not in actual_hashes:
            incorrect.append(
                f"{message_id}: expected={expected_hash}, "
                f"actual={sorted(actual_hashes) or 'no commit found'}"
            )

    if incorrect:
        warnings.warn(UserWarning(
            f"{len(incorrect)} incorrect message-commit associations:\n"
            + "\n".join(incorrect),
        ))
