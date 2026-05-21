CREATE TABLE IF NOT EXISTS chapters (
    chapter_id INT  NOT NULL,
    title      TEXT NOT NULL,
    page_start INT  NOT NULL,
    page_end   INT  NOT NULL,

    PRIMARY KEY (chapter_id)
);

CREATE TABLE IF NOT EXISTS highlights (
    highlight_id UUID      NOT NULL,
    chapter_id   INT       NOT NULL,
    page         INT       NOT NULL,
    text         TEXT      NOT NULL,
    note         TEXT      NULL,
    date_created TIMESTAMP NOT NULL,

    PRIMARY KEY (highlight_id),
    FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
);

CREATE TABLE IF NOT EXISTS bookmarks (
    bookmark_id  UUID      NOT NULL,
    chapter_id   INT       NOT NULL,
    page         INT       NOT NULL,
    label        TEXT      NULL,
    date_created TIMESTAMP NOT NULL,

    PRIMARY KEY (bookmark_id),
    FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
);
