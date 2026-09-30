# Agent Operating Rules: Token Discipline

These rules apply to every task in this repo. Follow them strictly. When a rule conflicts with an urge to be "thorough", the rule wins.

## 1. Scope
- Do exactly what was asked. Nothing more.
- Do not refactor, rename, reformat, add features, or "improve" code outside the request.
- Do not fix unrelated issues you notice. List them in one line at the end under "Noticed (not changed)".
- If the request is ambiguous and a wrong guess would waste significant work, ask ONE short question before starting. Otherwise, pick the most reasonable reading, state it in one line, and proceed.

## 2. Plan before acting
- Before any multi-step task, write a plan of at most 5 bullets: files to touch, change to make, how to verify.
- Stick to the plan. If you need to deviate, say why in one line.
- If the task needs more than ~10 file edits or touches unfamiliar systems, stop after the plan and wait for approval.

## 3. Reading and searching
- Search before reading. Use Grep/Glob to locate the exact file and lines, then read only the relevant range.
- Do not read whole large files, lockfiles, generated files, build output, `node_modules`, `dist`, `.git`, vendored code, or logs unless explicitly asked.
- Do not re-read a file you just read or edited unless it changed externally.
- Do not explore the codebase "to get context" beyond what the task requires.

## 4. Loop and runaway prevention (hard limits)
- **Max 3 attempts** at the same fix. If the same error or test failure persists after 3 attempts, STOP. Report: what you tried, the exact error, your best hypothesis. Do not try a 4th variation.
- Never retry an identical command expecting a different result.
- Never run commands that don't terminate on their own (dev servers, `--watch`, `tail -f`, interactive prompts) in the foreground. Use timeouts or background them, and stop them when done.
- Run the narrowest test possible (single file or single test), not the full suite, unless asked or unless finishing.
- If you notice yourself undoing a previous change, stop and reassess instead of continuing to cycle.
- Do not spawn subagents unless the user asks.

## 5. Editing
- Make minimal, targeted edits. Prefer editing existing code over rewriting files.
- Never rewrite an entire file to change a few lines.
- Do not add comments, docstrings, type annotations, or tests that weren't requested.
- Do not create new files (docs, READMEs, helpers, examples) unless required by the task.

## 6. Verification
- Verify once, at the end, with the cheapest check that proves the change works (targeted test, type check, or build of the affected package).
- If verification fails, that counts toward the 3-attempt limit.

## 7. Output
- Be terse. No preamble, no restating the request, no narrating each step.
- Do not paste large code blocks or file contents back into chat; the diff is enough.
- Final report format:
  - **Done:** one to three lines on what changed
  - **Files:** list of files touched
  - **Verified:** how
  - **Noticed (not changed):** optional, one line each
  - **Blocked:** only if stopped early, with the reason

## 8. When to stop and ask
Stop and ask instead of continuing if:
- You hit the 3-attempt limit.
- The fix requires changes outside the requested scope.
- You need to delete files, change dependencies, alter DB schemas/migrations, or touch config/CI.
- The task turns out to be significantly larger than it first appeared.
