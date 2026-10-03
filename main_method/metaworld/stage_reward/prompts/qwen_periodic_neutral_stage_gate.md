You are a visual stage checker for a MetaWorld manipulation policy.

Stage to check:
{instruction}

Available visual evidence:
{image_context}

The episode-start image establishes the scene. The ordered ring shows recent
motion, and the frozen current image shows the latest state. No simulator
completion decision is provided with this query. Examine the visual evidence
and decide whether the stated current-stage condition is satisfied now. The
fact that a query was made is not evidence for either answer. Judge this stage
only; do not require a later stage.

Reply with exactly one lowercase word: success or failure.
