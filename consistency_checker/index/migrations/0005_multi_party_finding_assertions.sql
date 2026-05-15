-- Join table giving multi_party_findings a proper FK into assertions.
-- The JSON list in `assertion_ids_json` stays for fast read of the verdict
-- payload; this side table is the referential-integrity surface so deleting
-- an assertion can't silently orphan a multi-party finding (pair `findings`
-- has the same guarantee via direct FK columns; ADR-0006 keeps the schemas
-- decoupled, but the canonical-store invariant has to hold for both).

CREATE TABLE IF NOT EXISTS multi_party_finding_assertions (
    finding_id TEXT NOT NULL REFERENCES multi_party_findings(finding_id) ON DELETE CASCADE,
    assertion_id TEXT NOT NULL REFERENCES assertions(assertion_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (finding_id, assertion_id)
);

CREATE INDEX IF NOT EXISTS idx_mpfa_finding ON multi_party_finding_assertions(finding_id);
CREATE INDEX IF NOT EXISTS idx_mpfa_assertion ON multi_party_finding_assertions(assertion_id);
