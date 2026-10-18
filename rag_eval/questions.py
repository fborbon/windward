"""The 15-question battery behind the RAG Evaluation tab (README §17): one question per
capability the /ask agent is meant to cover, each with the tool(s) that should answer it and
the facts a correct answer must contain.

`expect` holds regexes checked against the answer (case-insensitive), `forbid` ones it must not
match (a contradiction the expected facts alone would let through); `sources` the tool
sources (agents/qa_agent._run_tool) that can answer it, at least one of which must be used (an empty
list: no tool is needed, as for q33, whose answer is in the agent's system prompt). The reference values come from the data
behind each tool, noted in `truth`, not from the agent, so a wrong answer can't grade itself.
"""

# Set 1 (q01-q15) was the first battery; set 2 (q16-q35) was added afterwards to test the fixes
# on questions they weren't tuned on, including traps: a turbine that doesn't exist, a result that
# was never computed, and a farm whose export can't identify forced outages.
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
        "expect": [r"82\.4", r"\bmatch|same|agree|consistent"],
        "forbid": [r"does not match|doesn.t match|mismatch|differs"],
        "truth": "Datasheet p.6 says 82.4 m; data_sources/farms.py used 82.0 m until 2026-10-18 and now uses 82.4 m (match).",
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
        "expect": [r"Betz", r"0\.5937|0\.594", r"HillOfTowie[ _]?18|T18|turbine 18", r"exceed|above|over|higher than",
                   r"anemometer|measurement|impossible|artifact|artefact"],
        "forbid": [r"(does not|doesn.t|did not|never) (exceed|go above|reach)|below the Betz"],
        "truth": "rank_turbines('peak_cp'): HillOfTowie_18's peak Cp 0.5937 (0.5995 before the 82.4 m rotor fix) is above "
                 "the 0.5926 Betz limit, which is physically impossible: a measurement artifact (anemometer reading low).",
    },

    # ---- set 2 ----
    {
        "id": "q16", "farm": "hill_of_towie", "capability": "SCADA history",
        "question": "What was the farm's best single day of production, and how much energy did it produce?",
        "sources": ["farm-facts"],
        "expect": [r"2024-02-02|2 February 2024|February 2, 2024|Feb(ruary)? 2,? 2024", r"1,?057\.9|1\.06\s*GWh"],
        "truth": "farm_facts: best day 2024-02-02, 1,057.9 MWh (SCADA covers Jan-Aug 2024 only).",
    },
    {
        "id": "q17", "farm": "penmanshiel", "capability": "SCADA history",
        "question": "Which calendar year produced the most energy, and how much?",
        "sources": ["farm-facts"],
        "expect": [r"2017", r"84\.8"],
        "forbid": [r"2016[^.]*(most|highest)"],
        "truth": "farm_facts: 2017, 84.83 GWh (SCADA coverage 99.7%); 2016 and 2021 are partial years.",
    },
    {
        "id": "q18", "farm": "kelmarsh", "capability": "Farm metadata",
        "question": "Do all turbines have the same hub height? If not, which ones differ?",
        "sources": ["farm-facts"],
        "expect": [r"68\.5", r"78\.5", r"(Kelmarsh[ _]?|turbines? |T)3\b", r"\b6\b"],
        "truth": "Kelmarsh_WT_static.csv: Kelmarsh 3 and 6 at 68.5 m, Kelmarsh 1, 2, 4, 5 at 78.5 m.",
    },
    {
        "id": "q19", "farm": "hill_of_towie", "capability": "Farm metadata",
        "question": "When was the AeroUp blade upgrade installed on the turbines?",
        "sources": ["farm-facts"],
        "expect": [r"2021", r"2023"],
        "truth": "Hill_of_Towie_AeroUp_install_dates.csv: works from 2021-09-23 to 2023-05-26 across all 21 turbines.",
    },
    {
        "id": "q20", "farm": "kelmarsh", "capability": "OpenOA electrical losses",
        "question": "How much energy is lost between the turbines and the substation meter?",
        "sources": ["operational-assessment"],
        "expect": [r"0\.86|0\.859"],
        "truth": "OpenOA ElectricalLosses: 0.859% mean (5th-95th percentile -0.31% to 2.01%), 2016-05 to 2021-06.",
    },
    {
        "id": "q21", "farm": "penmanshiel", "capability": "OpenOA wake ranking",
        "question": "Which turbine loses the most energy to wakes from its neighbours?",
        "sources": ["turbine-ranking", "operational-assessment"],
        "expect": [r"(Penmanshiel[ _]?|turbine )10\b", r"18\.8"],
        "truth": "OpenOA WakeLosses long-term: Penmanshiel_10 18.82%, then Penmanshiel_09 18.49%.",
    },
    {
        "id": "q22", "farm": "hill_of_towie", "capability": "OpenOA AEP (not computed)",
        "question": "What is the long-term P50 annual energy production?",
        "sources": ["operational-assessment"],
        "expect": [r"not (computed|calculated|available|estimated)|skipped|no (long-term|P50)|isn.t available|wasn.t (computed|calculated)"],
        "forbid": [r"P50[^.]{0,40}\d+(\.\d+)?\s*GWh"],
        "truth": "Not computed: the export has no availability/curtailment channel and only one year of data (§15.5).",
    },
    {
        "id": "q23", "farm": "kelmarsh", "capability": "OpenOA AEP uncertainty",
        "question": "How uncertain is the long-term energy estimate, and which component contributes most?",
        "sources": ["operational-assessment"],
        "expect": [r"1\.17|1\.169", r"regression"],
        "truth": "Monthly linear AEP uncertainty 1.169%; largest component regression (0.702%), then reanalysis product (0.508%).",
    },
    {
        "id": "q24", "farm": "hill_of_towie", "capability": "Datasheet PDF",
        "question": "What is the turbine's rotor speed range and its cut-in wind speed?",
        "sources": ["technical-docs"],
        "expect": [r"6\s*(-|–|to)\s*18", r"3\s*(-|–|to)\s*5\s*m/s", r"p\.\s?[56]\b"],
        "truth": "Datasheet p.6: rotor speed 6-18 rpm, cut-in 3-5 m/s (also p.5).",
    },
    {
        "id": "q25", "farm": "hill_of_towie", "capability": "Datasheet PDF",
        "question": "What type of generator does the turbine use, and how is it cooled?",
        "sources": ["technical-docs"],
        "expect": [r"asynchronous", r"heat exchanger"],
        "truth": "Datasheet p.4: fully-enclosed asynchronous squirrel-cage generator, air-to-air heat exchanger; p.6: asynchronous, integrated heat exchanger.",
    },
    {
        "id": "q26", "farm": "penmanshiel", "capability": "Manufacturer PDF",
        "question": "What output does the EnergyUp upgrade allow for an MM82 turbine in a 60 Hz market?",
        "sources": ["technical-docs"],
        "expect": [r"2\.125\s*MW|2,125\s*kW"],
        "forbid": [r"60 Hz[^.]{0,30}2\.2\s*MW"],
        "truth": "EnergyUp sheet p.1: 2.125 MW for the MM82 and MM92 (60 Hz market); 2.2 MW is the 50 Hz figure.",
    },
    {
        "id": "q27", "farm": "kelmarsh", "capability": "Tender PDF",
        "question": "What rules did the Hollandse Kust (zuid) permit set to protect bats?",
        "sources": ["technical-docs"],
        "expect": [r"5\.0\s*m/s|5\s*m/s", r"15 August", r"30 September", r"sunset"],
        "truth": "HKZ PSD p.33: cut-in 5.0 m/s at axis height 15 Aug-30 Sep, 1 h after sunset to 2 h before sunrise; below it under 1 rpm.",
    },
    {
        "id": "q28", "farm": "hill_of_towie", "capability": "Tender PDF",
        "question": "What are the 50-year design wave height and wind speed at the Hollandse Kust (zuid) site?",
        "sources": ["technical-docs"],
        "expect": [r"7\.0\s*m", r"40\.4\s*(m/s|met(er|re)s per second)", r"p\.\s?23\b"],
        "truth": "HKZ PSD p.23: 50-year significant wave height 7.0 m, hourly wind speed 40.4 m/s at 100 m MSL.",
    },
    {
        "id": "q29", "farm": "penmanshiel", "capability": "Tender PDF",
        "question": "What water depths and mean wind speed were reported for the Borssele wind farm sites?",
        "sources": ["technical-docs"],
        "expect": [r"Borssele", r"9\.6", r"17\.8|14\.0|16\s*(-|–|to)\s*38"],
        "truth": "Borssele PSD: zone depth about 16-38 m (p.13); BWFS I 17.8-39.7 m and II 14.0-38.5 m LAT (p.29); mean wind 9.6 ± 0.5 m/s at 100 m (p.37).",
    },
    {
        "id": "q30", "farm": "kelmarsh", "capability": "Turbine ranking",
        "question": "Which turbine has the highest capacity factor, and what is it?",
        "sources": ["turbine-ranking", "current-analysis"],
        "expect": [r"(Kelmarsh[ _]?|turbine )2\b", r"0\.29\d\d|0\.30\b|30(\.0)?\s*%|29\.99"],
        "truth": "Live analysis (2016): Kelmarsh_2, capacity factor 0.2999, then Kelmarsh_4 at 0.2663.",
    },
    {
        "id": "q31", "farm": "penmanshiel", "capability": "Status-log ranking",
        "question": "Which turbine had the most maintenance stops?",
        "sources": ["turbine-ranking"],
        "expect": [r"(Penmanshiel[ _]?|turbine )0?1\b", r"\b37\b"],
        "truth": "rank_turbines('maintenance_events'): Penmanshiel_01 with 37 stops, then Penmanshiel_04 (35).",
    },
    {
        "id": "q32", "farm": "hill_of_towie", "capability": "Status-log limit",
        "question": "Which turbine had the most forced-outage hours?",
        "sources": ["turbine-ranking", "turbine-details"],
        "expect": [r"not (identif|availab|possible|recorded|label)|can.t be identified|cannot be identified|doesn.t (label|identify|record)|does not (label|identify|record)|no forced"],
        "forbid": [r"HillOfTowie[ _]?\d+[^.]{0,60}most forced"],
        "truth": "Not answerable: the RES alarm export doesn't label forced outages (agents/turbine_tools.py), so none are ranked.",
    },
    {
        "id": "q33", "farm": "kelmarsh", "capability": "False premise",
        "question": "Why did Kelmarsh 7 stop for so long in 2019?",
        "sources": [],
        "expect": [r"(no|not a|isn.t a|doesn.t have a|does not have a|there is no) (turbine (named |called )?)?(Kelmarsh[ _]?7|turbine 7|seventh)|six turbines|6 turbines|Kelmarsh_1 to Kelmarsh_6|only (has )?(six|6)"],
        "truth": "Kelmarsh has six turbines (Kelmarsh 1-6): there is no Kelmarsh 7.",
    },
    {
        "id": "q34", "farm": "penmanshiel", "capability": "Forecast vs actual",
        "question": "How did the farm's actual production compare with the model's prediction?",
        "sources": ["current-analysis"],
        "expect": [r"above|higher|more than|exceed", r"\b[1-9]\d(\.\d+)?\s*%"],
        "forbid": [r"0\.50\d?\s*%", r"(lower|less) than the (model.s )?predict|below (the )?predict"],
        "truth": "Live analysis comparison_summary: actual slightly above predicted (5.654 MW vs 5.653-5.654 MW) over 5,055 hours, "
                 "mean absolute hourly error around 50% (50.4% on one run, 52.3% after a redeploy recomputed the analysis).",
    },
    {
        "id": "q35", "farm": "hill_of_towie", "capability": "Anomaly detection",
        "question": "Which turbines are suspected of an anemometer calibration problem, and why?",
        "sources": ["current-analysis", "turbine-details"],
        "expect": [r"HillOfTowie[ _]?18|T18|turbine 18", r"0\.31|0\.306", r"anemometer|wind speed"],
        "forbid": [r"HillOfTowie[ _]?16|T16\b|turbine 16"],
        "truth": "Live analysis anomalies (rotor 82.4 m): only HillOfTowie_18, peak Cp above Betz with its power curve displaced +0.31 m/s from the farm mean. HillOfTowie_16 was also flagged until the rotor fix.",
    },
]
