INSERT INTO users (user_id, name, email, roles, password_hash, department, enabled, date_created, date_updated)
VALUES 
    ('4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', 'Luis Rodriguez', 'luis.rodriguez@example.com', '{admin}', 'hashed_password', NULL, TRUE, NOW(), NOW()),
    ('cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', 'Sofia Sanchez', 'sofia.sanchez@example.com', '{user}', 'hashed_password', NULL, TRUE, NOW(), NOW()),
    ('be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', 'Juan Hernandez', 'juan.hernandez@example.com', '{admin,user}', 'hashed_password', NULL, TRUE, NOW(), NOW()),
    ('f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'Maria Garcia', 'maria.garcia@example.com', '{user}', 'hashed_password', NULL, TRUE, NOW(), NOW()),
    ('a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'Elena Vasquez', 'elena.vasquez@example.com', '{admin,user}', 'hashed_password', NULL, TRUE, NOW(), NOW())
ON CONFLICT(user_id) DO NOTHING;

INSERT INTO products (product_id, user_id, name, cost, quantity, date_created, date_updated)
VALUES 
    ('1d4cbb14-c2b9-48a8-bd17-4dbf6e3d33eb', '4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', 'Smartwatch', 129.99, 10, NOW(), NOW()),
    ('2c58ad31-e42c-49ac-bdf1-fd34ec6e6bde', '4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', 'Wireless Headphones', 79.99, 20, NOW(), NOW()),
    ('2ca65ad8-cba1-49c3-a1df-fd3429c86962', '4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', 'Portable Charger', 39.99, 15, NOW(), NOW()),
    ('2ca65ad9-cba2-49c3-a1df-fd34ec6e6bde', '4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', 'Power Bank Case', 29.99, 25, NOW(), NOW()),
    ('2ca65ad7-cba1-49c3-a1df-fd3429c86962', '4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', 'Smartphone Stand', 19.99, 30, NOW(), NOW())
ON CONFLICT(product_id) DO NOTHING;

INSERT INTO products (product_id, user_id, name, cost, quantity, date_created, date_updated)
VALUES 
    ('e43a77ba-f2c4-4b9f-aec6-bcd0a7ca3d14', 'cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', 'Designer Watch', 199.99, 8, NOW(), NOW()),
    ('a51a74b3-e7ec-41ea-92e5-cdcf54e85c21', 'cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', 'Premium Headphones', 99.99, 18, NOW(), NOW()),
    ('a51a74b4-e7ec-41ea-92e5-cdcf54e85c21', 'cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', 'Wireless Earbuds', 69.99, 22, NOW(), NOW()),
    ('a51a74b5-e7ec-41ea-92e5-cdcf54e85c21', 'cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', 'Portable Power Bank', 49.99, 25, NOW(), NOW()),
    ('a51a74b2-e7ec-41ea-92e5-cdcf54e85c21', 'cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', 'Smartphone Accessory Kit', 39.99, 28, NOW(), NOW())
ON CONFLICT(product_id) DO NOTHING;

INSERT INTO products (product_id, user_id, name, cost, quantity, date_created, date_updated)
VALUES 
    ('f79e9d31-b5aa-450b-a8dd-f3efeb7f9a16', 'be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', 'Smartphone Case', 14.99, 32, NOW(), NOW()),
    ('d4e43e59-cbb5-46aa-a54d-b35f33dca8dd', 'be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', 'Wireless Earbud Case', 24.99, 20, NOW(), NOW()),
    ('c94a5e83-40db-44e0-82f7-fd4ff4ea3b35', 'be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', 'Power Bank Case', 34.99, 18, NOW(), NOW()),
    ('a54da56a-e4db-47e5-b4cd-74f0fa29aa11', 'be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', 'Smartwatch Band', 39.99, 15, NOW(), NOW()),
    ('7c85f6d8-e63e-4494-b1d0-5dc7bceef6ff', 'be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', 'Phone Stand', 29.99, 12, NOW(), NOW())
ON CONFLICT(product_id) DO NOTHING;

INSERT INTO products (product_id, user_id, name, cost, quantity, date_created, date_updated)
VALUES 
    ('f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'Premium Headphones', 99.99, 30, NOW(), NOW()),
    ('f7d6f4e6-a2db-44d3-bb52-e45ec8e4ff11', 'f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'Wireless Earbuds', 69.99, 25, NOW(), NOW()),
    ('f7d6f4e7-a2db-44d3-bb53-e45ec8e4ff12', 'f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'Portable Power Bank', 49.99, 20, NOW(), NOW()),
    ('f7d6f4e8-a2db-44d3-bb54-e45ec8e4ff13', 'f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'Smartwatch Band', 39.99, 15, NOW(), NOW()),
    ('f7d6f4e9-a2db-44d3-bb55-e45ec8e4ff14', 'f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', 'Phone Stand', 29.99, 10, NOW(), NOW())
ON CONFLICT(product_id) DO NOTHING;

INSERT INTO products (product_id, user_id, name, cost, quantity, date_created, date_updated)
VALUES 
    ('a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'Gaming Keyboard', 129.99, 40, NOW(), NOW()),
    ('e54ba7ca-1d81-48da-bbb4-d7cefe8a31ad', 'a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'Wireless Gaming Mouse', 89.99, 35, NOW(), NOW()),
    ('d87d1b55-f38e-4894-bdf5-ea3ca3b15b22', 'a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'External Hard Drive', 59.99, 25, NOW(), NOW()),
    ('67e24e42-d8df-4bb4-b6d1-fa33bcdaaa11', 'a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'Smartphone Case', 19.99, 15, NOW(), NOW()),
    ('3d0d85db-b1ec-4ff5-be33-c5daa60dfbf3', 'a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', 'Gaming Headset', 79.99, 10, NOW(), NOW())
ON CONFLICT(product_id) DO NOTHING;

INSERT INTO homes (home_id, type, user_id, address_1, city, state, zip_code, country, date_created, date_updated)
VALUES 
    ('faba52c5-7059-4811-abf7-856ee64a22fa', 'single-family', '4bce0a9f-8e95-43f7-ba47-d3daeb5dbb83', '123 Main St', 'New York City', 'NY', '10001', 'US', NOW(), NOW()),
    ('d18e1128-8528-4079-89ce-a39d100ba1c1', 'apartment',     'cdedc2d1-a8df-4ae6-9e14-befaa0f5d7ff', '456 Broadway', 'Los Angeles', 'CA', '90012', 'US', NOW(), NOW()),
    ('c01e6874-a0b4-4710-957a-6e050e2068ce', 'condo',         'be3ac8b6-d66c-42ad-a1df-c77ab2a90ae9', '789 Park Ave', 'Chicago', 'IL', '60010', 'US', NOW(), NOW()),
    ('70605636-edaa-440f-9b90-e8b1bf2e935a', 'multi-family',  'f7d6f4e5-a2db-44d3-bb51-e45ec8e4ff1c', '901 Oak St', 'Houston', 'TX', '77001', 'US', NOW(), NOW()),
    ('ed6b3228-c191-4745-bb6e-4d2b4ba845b1', 'single-family', 'a11d0d34-8dd9-41c7-af2c-a6b3f5c6ab27', '234 Wall St', 'San Francisco', 'CA', '94102', 'US', NOW(), NOW())
ON CONFLICT(home_id) DO NOTHING;
