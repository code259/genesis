You are preparing a Python execution plan for a single research task that requires concrete artifacts.

Return JSON only with this schema:
{
  "python_code": "full runnable python script",
  "expected_artifacts": [
    {"path": "relative/path/from/task_artifact_dir", "kind": "csv|png|json|txt"}
  ],
  "notes": ["short notes about assumptions or limitations"]
}

Rules:
- Use only Python standard library plus numpy/scipy if needed.
- Write all outputs under the directory given by the GENESIS_ARTIFACT_DIR environment variable.
- Write a JSON summary to the file path in GENESIS_RESULTS_PATH.
- If a PNG is needed but you cannot rely on plotting libraries, create it with standard-library-safe output.
- Do not wrap the JSON in markdown fences.
