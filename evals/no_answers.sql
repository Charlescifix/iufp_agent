-- Live questions the bot couldn't answer from IUFP's guides, newest first.
-- Recurring ones are gaps in the knowledge base (or retrieval misses): check each
-- against evals/cases.json and add a case for any worth keeping.
SELECT created_at, session_id, user_message, bot_response
FROM chat_messages
WHERE bot_response ILIKE 'I don''t have that information%'
   OR bot_response ILIKE 'I don’t have that information%'
ORDER BY created_at DESC
LIMIT 100;
