INSERT INTO chapters (chapter_id, title, page_start, page_end)
VALUES
    ( 1, 'Introduction',          10,  24),
    ( 2, 'Language Mechanics',    25,  40),
    ( 3, 'Data Structures',       41,  66),
    ( 4, 'Decoupling',            67,  89),
    ( 5, 'Software Design',       90, 118),
    ( 6, 'Concurrency',          119, 150),
    ( 7, 'Testing',              151, 162),
    ( 8, 'Benchmarking',         163, 173),
    ( 9, 'Generics',             174, 199),
    (10, 'Profiling',            200, 240)
ON CONFLICT(chapter_id) DO NOTHING;

INSERT INTO highlights (highlight_id, chapter_id, page, text, note, date_created)
VALUES
    ('11111111-0000-0000-0000-000000000001', 2,  32, 'Pointers serve the purpose of sharing.',                             'core idea',         NOW()),
    ('11111111-0000-0000-0000-000000000002', 2,  34, 'Escape analysis determines where values live: stack or heap.',       NULL,                NOW()),
    ('11111111-0000-0000-0000-000000000003', 3,  50, 'Slices are dynamically sized views over an underlying array.',       NULL,                NOW()),
    ('11111111-0000-0000-0000-000000000004', 3,  62, 'UTF-8 encoding is the basis of Go string handling.',                  NULL,                NOW()),
    ('11111111-0000-0000-0000-000000000005', 4,  76, 'Interfaces are valueless.',                                            'fundamental rule',  NOW()),
    ('11111111-0000-0000-0000-000000000006', 4,  79, 'Polymorphism in Go is achieved through interfaces.',                  NULL,                NOW()),
    ('11111111-0000-0000-0000-000000000007', 5, 105, 'Interface pollution is a real risk; design for concrete needs first.', NULL,               NOW()),
    ('11111111-0000-0000-0000-000000000008', 6, 121, 'Concurrency is about managing many tasks; parallelism is about doing them at once.', NULL, NOW()),
    ('11111111-0000-0000-0000-000000000009', 6, 127, 'Data races corrupt program state in subtle ways.',                   NULL,                NOW()),
    ('11111111-0000-0000-0000-000000000010', 6, 137, 'Channel semantics: signaling, not data sharing.',                     'remember this',     NOW()),
    ('11111111-0000-0000-0000-000000000011', 7, 153, 'Table tests are the canonical Go testing pattern.',                   NULL,                NOW()),
    ('11111111-0000-0000-0000-000000000012', 9, 175, 'Generics in Go are constrained by interfaces, not class hierarchies.', NULL,               NOW())
ON CONFLICT(highlight_id) DO NOTHING;

INSERT INTO bookmarks (bookmark_id, chapter_id, page, label, date_created)
VALUES
    ('22222222-0000-0000-0000-000000000001', 4,  76, 'Re-read interfaces section', NOW()),
    ('22222222-0000-0000-0000-000000000002', 6, 137, 'Channel patterns',            NOW()),
    ('22222222-0000-0000-0000-000000000003', 6, 146, 'Fan out/in semaphore',        NOW()),
    ('22222222-0000-0000-0000-000000000004', 9, 180, 'Behavior as constraint',      NOW()),
    ('22222222-0000-0000-0000-000000000005', 10, 200, 'Start profiling here',       NOW())
ON CONFLICT(bookmark_id) DO NOTHING;
