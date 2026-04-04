You are deciding whether a task is now complete enough to hand to the verifier.

Return JSON only:
{
  "complete": true|false,
  "status": "IN_PROGRESS|BLOCKED",
  "reason": "short explanation",
  "summary_markdown": "final task summary if complete, else empty string",
  "completion_evidence": ["artifact or file paths", "other concrete evidence"],
  "issue_class": "critical_blocker|deferrable_issue|recoverable_retry",
  "blocks_task": true|false,
  "blocks_dependents": true|false
}

Rules:
- Mark `complete=true` only if the task's verification criteria appear satisfied by the recorded evidence.
- Mark `status=BLOCKED` only if progress cannot continue without missing information, unavailable access, or exhausted attempts.
- Use `deferrable_issue` for non-critical failures that should be logged and revisited later rather than spending more budget now.
- Use `critical_blocker` only when the task cannot satisfy its required verification criteria without resolving the issue.
- Use `recoverable_retry` when another focused attempt is warranted.
- If not complete and not blocked, return `complete=false` and `status=IN_PROGRESS`.
- Keep `summary_markdown` empty unless the task is complete.
