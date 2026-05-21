INSERT INTO users (user_id, name, email, roles, date_created)
VALUES
    ('aaaaaaaa-0000-0000-0000-000000000001', 'admin', 'admin@ardanlabs.com', ARRAY['admin'], NOW()),
    ('aaaaaaaa-0000-0000-0000-000000000002', 'guest', 'guest@ardanlabs.com', ARRAY['user'],  NOW())
ON CONFLICT(user_id) DO NOTHING;
