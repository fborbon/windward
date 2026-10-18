"""The 15-question battery behind the RAG Evaluation tab (README §17): one question per
capability the /ask agent is meant to cover, each with the tool(s) that should answer it and
the facts a correct answer must contain.

`expect` holds regexes checked against the answer (case-insensitive), `forbid` ones it must not
match (a contradiction the expected facts alone would let through); `sources` the tool
sources (agents/qa_agent._run_tool) that can answer it, at least one of which must be used. The reference values come from the data
behind each tool, noted in `truth`, not from the agent, so a wrong answer can't grade itself.
"""

QUESTIONS = [
    {
        "id": "q01", "farm": "kelmarsh", "capability": "SCADA history",
        "question": "What is the maximum historical power production of the wind farm?",
        "sources": ["farm-facts"],
        "expect": [r"12[.,]36\s*MW|12,36\d\s*kW", r"2018"],
        "truth": "farm_facts_payload.json: highest 10-minute farm output in the 2016-2021 SCADA is 12.36 MW "
                 "(installed 12.3 MW) on 2018-01-04 17:40; best day 287.0 MWh, best full year 2020 (36.35 GWh).",
    },
    {
        "id": "q02", "farm": "hill_of_towie", "capability": "Datasheet PDF",
        "question": "How many different hub heights, rotor sizes and turbulence categories does the turbine manufacturer offer for this model?",
        "sources": ["technical-docs"],
        "expect": [r"82\.4", r"80\s*m", r"site.specific", r"\bIA\b|\(A\)|category:?\s*A\b"],
        "truth": "Siemens SWT-2.3-82 VS datasheet: one rotor (82.4 m, p.6), hub height '80 m or site-specific' (p.6), "
                 "IEC Class IA (p.4); standard and IEC versions differ in max gust (55 / 70 m/s).",
    },
    {
        "id": "q03", "farm": "kelmarsh", "capability": "Farm metadata",
        "question": "When was the wind farm built and when will it be decommissioned?",
        "sources": ["farm-facts"],
        "expect": [r"2016", r"not (stated|in|available|known|recorded|published|provided|specified)|no (decommission|date|information|record)|doesn.t (state|include)|does not (state|include)"],
        "truth": "Kelmarsh_WT_static.csv: commercial operations 15/04/2016 for all 6 turbines. No decommissioning date in any source.",
    },
    {
        "id": "q04", "farm": "penmanshiel", "capability": "Turbine ranking",
        "question": "Which is the best and which is the worst performing turbine?",
        "sources": ["turbine-ranking", "current-analysis"],
        "expect": [r"Penmanshiel[ _]?\d+", r"capacity factor"],
        "truth": "rank_turbines('capacity_factor'): top and bottom of the ranking.",
    },
    {
        "id": "q05", "farm": "kelmarsh", "capability": "OpenOA wake losses",
        "question": "What is the global wake effect in % of the annual energy yield?",
        "sources": ["operational-assessment"],
        "expect": [r"7\.7[78]|7\.8\b|8\.2[56]|8\.3\b"],
        "truth": "OpenOA WakeLosses: plant 7.78% long-term corrected (8.26% over the period of record).",
    },
    {
        "id": "q06", "farm": "hill_of_towie", "capability": "Datasheet PDF",
        "question": "At what wind speed does the turbine shut down, and what kind of gearbox does it have?",
        "sources": ["technical-docs"],
        "expect": [r"25\s*m/s", r"three.stage|3.stage", r"planetary", r"p\.\s?[456]\b|pages? [456]"],
        "truth": "Datasheet p.5/p.6: cut-out 25 m/s; p.4: three-stage planetary-helical gearbox.",
    },
    {
        "id": "q07", "farm": "penmanshiel", "capability": "OpenOA AEP",
        "question": "What are the long-term P50 and P90 annual energy production estimates?",
        "sources": ["operational-assessment"],
        "expect": [r"76\.[3-6]", r"74\.[7-8]"],
        "truth": "OpenOA MonteCarloAEP (monthly linear): P50 76.3 GWh, P90 74.8 GWh.",
    },
    {
        "id": "q08", "farm": "kelmarsh", "capability": "OpenOA yaw",
        "question": "Which turbine has the largest static yaw misalignment, and how large is it?",
        "sources": ["turbine-ranking", "operational-assessment"],
        "expect": [r"(Kelmarsh[ _]?|turbine )4\b", r"18\.47|18\.5", r"inconclusive|not reliable|unreliable"],
        "truth": "OpenOA StaticYawMisalignment: every Kelmarsh estimate is inconclusive (per-bin estimates spread "
                 "14-34 deg); the largest raw value is Kelmarsh 4 at 18.47 deg, not a reliable figure.",
    },
    {
        "id": "q09", "farm": "penmanshiel", "capability": "Status-log ranking",
        "question": "Which turbine had the most forced-outage hours, and what were the main causes?",
        "sources": ["turbine-ranking", "turbine-details"],
        "expect": [r"Penmanshiel[ _]?11", r"726\.5", r"Emergency stop base box"],
        "truth": "rank_turbines('forced_outage_hours'): Penmanshiel_11, 726.5 h; get_turbine_details: top cause "
                 "'Emergency stop base box' (2 events, 716.2 h), then 'Externally stopped' (3.7 h).",
    },
    {
        "id": "q10", "farm": "kelmarsh", "capability": "Fault-log RAG",
        "question": "What were the longest downtime events and what caused them?",
        "sources": ["maintenance-docs"],
        "expect": [r"(Kelmarsh[ _]?|turbine )1\b", r"548\.2", r"brake"],
        "truth": "rag/incident_corpus.py: the longest real status-log stop is Kelmarsh 1, 548.2 h, "
                 "'Timeout brake closed' (2016-12-17).",
    },
    {
        "id": "q11", "farm": "hill_of_towie", "capability": "Datasheet cross-check",
        "question": "Does the rotor diameter Windward uses match the manufacturer's datasheet?",
        "sources": ["technical-docs"],
        "expect": [r"82\.4", r"82(\.0)?\b", r"differ|does not match|doesn.t match|mismatch|not match"],
        "truth": "Datasheet p.6 says 82.4 m; data_sources/farms.py uses 82.0 m.",
    },
    {
        "id": "q12", "farm": "kelmarsh", "capability": "Manufacturer PDF",
        "question": "What upgrade does the manufacturer offer to raise this turbine's energy output, and by how much?",
        "sources": ["technical-docs"],
        "expect": [r"EnergyUp|Full Power Curve", r"5\s*%", r"2\.2\s*MW", r"Bachmann"],
        "truth": "Siemens Gamesa EnergyUp sheet: software upgrade raising MM92 output to 2.2 MW (50 Hz), AEP up to "
                 "5%, only for turbines with a Bachmann controller.",
    },
    {
        "id": "q13", "farm": "hill_of_towie", "capability": "Leasing-round PDF",
        "question": "What does an applicant have to submit in the ScotWind leasing round?",
        "sources": ["technical-docs"],
        "expect": [r"ScotWind", r"p\.\s?\d"],
        "truth": "Crown Estate Scotland, ScotWind Leasing April 2019 update.",
    },
    {
        "id": "q14", "farm": "penmanshiel", "capability": "Tender PDF",
        "question": "What water depths and mean wind speed were reported for the Hollandse Kust (zuid) tender sites?",
        "sources": ["technical-docs"],
        "expect": [r"Hollandse Kust", r"9\.44", r"18\.[18]", r"27\.[89]|26\.6", r"p\.\s?(9|14)\b", r"p\.\s?21\b"],
        "truth": "RVO HKZ Project and Site Description: water depth 18.1-27.9 m LAT (Site I) and 18.8-26.6 m (Site II), "
                 "p.14; long-term mean wind speed 9.44 +/- 0.38 m/s at 100 m MSL, p.21.",
    },
    {
        "id": "q15", "farm": "hill_of_towie", "capability": "Betz-limit analysis",
        "question": "How close does the best turbine get to the Betz limit?",
        "sources": ["turbine-ranking", "current-analysis"],
        "expect": [r"Betz", r"0\.599", r"HillOfTowie[ _]?18|T18|turbine 18", r"exceed|above|over|higher than",
                   r"anemometer|measurement|impossible|artifact|artefact"],
        "forbid": [r"(does not|doesn.t|did not|never) (exceed|go above|reach)|below the Betz"],
        "truth": "rank_turbines('peak_cp'): HillOfTowie_18's peak Cp 0.5995 is above the 0.593 Betz limit, which is "
                 "physically impossible: a measurement artifact (anemometer reading low), not a near-Betz turbine.",
    },
]
