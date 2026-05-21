CREATE TABLE IF NOT EXISTS users (
    user_id      UUID        NOT NULL,
    name         TEXT UNIQUE NOT NULL,
    email        TEXT UNIQUE NOT NULL,
    roles        TEXT[]      NOT NULL,
    date_created TIMESTAMP   NOT NULL,

    PRIMARY KEY (user_id)
);
