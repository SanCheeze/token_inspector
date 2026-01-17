CREATE TABLE IF NOT EXISTS tokens (
    id SERIAL PRIMARY KEY,
    token TEXT UNIQUE NOT NULL,
    symbol TEXT,
    content JSONB,
    supply BIGINT,
    trades JSONB,
    trades_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wallets (
    id SERIAL PRIMARY KEY,
    token TEXT NOT NULL,
    wallet TEXT NOT NULL,
    pnl DOUBLE PRECISION,
    num_buys INTEGER,
    num_sells INTEGER,
    trades JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(token, wallet)
);

CREATE INDEX IF NOT EXISTS idx_wallets_token ON wallets(token);
CREATE INDEX IF NOT EXISTS idx_wallets_wallet ON wallets(wallet);
