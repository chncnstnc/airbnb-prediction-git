DROP TABLE IF EXISTS Listing;
DROP TABLE IF EXISTS Host;

CREATE TABLE Host (
    host_id INTEGER PRIMARY KEY,
    host_name TEXT
);

CREATE TABLE Listing (
    id INTEGER PRIMARY KEY,
    name TEXT,
    host_id INTEGER,
    neighbourhood TEXT,
    room_type TEXT,
    price REAL,
    minimum_nights INTEGER,
    number_of_reviews INTEGER,
    availability_365 INTEGER,
    FOREIGN KEY (host_id) REFERENCES Host(host_id)
);