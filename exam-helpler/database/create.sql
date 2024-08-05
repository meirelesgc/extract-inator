CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE TABLE IF NOT EXISTS exames (
    exames_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    descricao VARCHAR(200) NOT NULL,
    PRIMARY KEY (exames_id)
);