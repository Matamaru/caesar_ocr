-- Company universe schema (SQLite)

CREATE TABLE company (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  street TEXT,
  street_no TEXT,
  zipcode TEXT,
  town TEXT,
  ik TEXT
);

CREATE TABLE customer (
  id INTEGER PRIMARY KEY,
  company_id INTEGER NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  nationality TEXT NOT NULL,
  street TEXT,
  street_no TEXT,
  zipcode TEXT,
  town TEXT,
  FOREIGN KEY (company_id) REFERENCES company(id)
);

CREATE TABLE employee (
  id INTEGER PRIMARY KEY,
  company_id INTEGER NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  FOREIGN KEY (company_id) REFERENCES company(id)
);

CREATE TABLE service (
  id INTEGER PRIMARY KEY,
  code TEXT NOT NULL,
  legal_ref TEXT,
  name TEXT
);

CREATE TABLE customer_service (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  service_id INTEGER NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES customer(id),
  FOREIGN KEY (service_id) REFERENCES service(id)
);

CREATE TABLE customer_service_signature (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  service_id INTEGER NOT NULL,
  period_year INTEGER NOT NULL,
  period_month INTEGER NOT NULL,
  signed INTEGER NOT NULL,
  updated_at TEXT,
  FOREIGN KEY (customer_id) REFERENCES customer(id),
  FOREIGN KEY (service_id) REFERENCES service(id)
);

CREATE TABLE document_type (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  title TEXT
);

CREATE TABLE instance_type (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE instance (
  id INTEGER PRIMARY KEY,
  company_id INTEGER NOT NULL,
  customer_id INTEGER NOT NULL,
  instance_type_id INTEGER NOT NULL,
  acc_period_start TEXT,
  acc_period_end TEXT,
  created_at TEXT,
  FOREIGN KEY (company_id) REFERENCES company(id),
  FOREIGN KEY (customer_id) REFERENCES customer(id),
  FOREIGN KEY (instance_type_id) REFERENCES instance_type(id)
);

CREATE TABLE instance_service (
  id INTEGER PRIMARY KEY,
  instance_id INTEGER NOT NULL,
  service_id INTEGER NOT NULL,
  quantity REAL,
  unit_price REAL,
  total_price REAL,
  FOREIGN KEY (instance_id) REFERENCES instance(id),
  FOREIGN KEY (service_id) REFERENCES service(id)
);

CREATE TABLE document (
  id INTEGER PRIMARY KEY,
  company_id INTEGER NOT NULL,
  document_type_id INTEGER NOT NULL,
  title TEXT,
  customer_id INTEGER,
  employee_id INTEGER,
  instance_id INTEGER,
  created_at TEXT,
  file_path TEXT,
  FOREIGN KEY (company_id) REFERENCES company(id),
  FOREIGN KEY (document_type_id) REFERENCES document_type(id),
  FOREIGN KEY (customer_id) REFERENCES customer(id),
  FOREIGN KEY (employee_id) REFERENCES employee(id),
  FOREIGN KEY (instance_id) REFERENCES instance(id)
);

CREATE TABLE doc_label (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT
);

CREATE TABLE document_label_value (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  doc_label_id INTEGER NOT NULL,
  value TEXT,
  confidence REAL,
  page INTEGER,
  bbox TEXT,
  FOREIGN KEY (document_id) REFERENCES document(id),
  FOREIGN KEY (doc_label_id) REFERENCES doc_label(id)
);
