"""The written review of each answer in the battery: a verdict a human (here, the reviewer
reading the answer against `truth` in rag_eval/questions.py) gives the baseline and the final
answer, what went wrong and which change fixed it. The automatic grade in rag_eval/run.py is a
first pass; these verdicts are the ones the RAG Evaluation tab shows, because the regex grade got
two baseline answers wrong (q05's correct 7.778% failed an over-strict pattern, and q15's
"closest to the Betz limit" for a Cp above it passed).

verdict: correct | partial | wrong | refused. "baseline" is the live answer before the changes
(rag_eval/runs/baseline.json), "final" the live answer after they deployed (runs/final.json).
"""

REVIEW = {
    "q01": {
        "baseline": "wrong", "final": "correct",
        "diagnosis": "Answered 3.007 MW, a value from the 2016 analysis window: no tool held the multi-year SCADA history.",
        "fix": "New get_farm_facts tool: production records over the full 2016-2021 SCADA, exported to a small JSON (data_sources/farm_facts.py).",
    },
    "q02": {
        "baseline": "refused", "final": "correct",
        "diagnosis": "Retrieval matched the brochure's prose pages; the spec table (p.6) with the hub height never reached the model.",
        "fix": "The farm's own manufacturer documents are read whole (8 pages), plus prompt rules for 'which options' questions and the IEC class letter.",
    },
    "q03": {
        "baseline": "refused", "final": "correct",
        "diagnosis": "Said it couldn't know; the commissioning date was in the dataset's own turbine metadata, which no tool read.",
        "fix": "get_farm_facts returns the commercial operations date and says plainly that no source has a decommissioning date.",
    },
    "q04": {"baseline": "correct", "final": "correct", "diagnosis": "rank_turbines by capacity factor, quoted exactly.", "fix": None},
    "q05": {
        "baseline": "correct", "final": "correct",
        "diagnosis": "7.778% long-term corrected is right. The automatic grade failed it: the evaluator's pattern was wrong, not the answer.",
        "fix": "Grader pattern corrected. Later runs also said what lt_pct means instead of printing the field name.",
    },
    "q06": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Right values, but no page citation and padded with 'in line with standard practices'.",
        "fix": "The tool's verified page citations are re-attached when the agent's rewrite drops them.",
    },
    "q07": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Quoted the daily GAM method's P50/P90 (76.6/75.7 GWh) without saying it's the second of two methods.",
        "fix": "The OpenOA summary now names the headline method (monthly linear, P50 76.3 / P90 74.8 GWh).",
    },
    "q08": {
        "baseline": "wrong", "final": "correct",
        "diagnosis": "Named Kelmarsh 1. Every Kelmarsh yaw estimate is inconclusive, so the ranking dropped them all and came back empty.",
        "fix": "The ranking keeps inconclusive estimates, labeled and listed after conclusive ones. Kelmarsh 4 (18.47 deg) is the largest, and is unreliable.",
    },
    "q09": {
        "baseline": "correct", "final": "correct",
        "diagnosis": "Correct at baseline, but later runs showed the agent requesting the turbine's details in the same parallel tool call as the ranking, before it knew the turbine, and then inventing the causes.",
        "fix": "A ranking by outage hours now carries the top turbine's status log, and the agent is told never to fill a failed tool call.",
    },
    "q10": {"baseline": "correct", "final": "correct", "diagnosis": "Longest stops from the real status log, cited by event.", "fix": "A missing IEC category no longer prints as 'nan'."},
    "q11": {"baseline": "correct", "final": "correct",
            "diagnosis": "82.4 m datasheet vs 82.0 m configured, from the verified cross-check.",
            "fix": "The finding was acted on: farms.py now uses 82.4 m, so the answer is now 'match' (the reference changed with it)."},
    "q12": {
        "baseline": "partial", "final": "partial",
        "diagnosis": "Right upgrade, 2.2 MW and up to 5%, but uncited, and it drops the sheet's scope note (only turbines with a Bachmann controller).",
        "fix": "Citations now kept. The scope note is still dropped despite an explicit prompt rule: a remaining Nova Lite failure.",
    },
    "q13": {
        "baseline": "partial", "final": "partial",
        "diagnosis": "Right six areas, but leaked a '</thinking>' tag and a sentence of planning. Later runs also padded each item with invented explanations, or returned an empty answer.",
        "fix": "Reasoning tags stripped, a rule against explanations the page doesn't give, and one retry on an empty reply. "
               "It held on one final run, but the re-run after set 2 padded the six areas again ('A clear and detailed "
               "description of the proposed project...'): the padding comes from the tool's own answer, and the "
               "no-padding rule doesn't hold reliably on Nova Lite.",
    },
    "q14": {
        "baseline": "refused", "final": "correct",
        "diagnosis": "The depth passage was retrieved and Nova still refused. The 9.44 m/s sentence sat in the chunk next to the one that matched. Once answered, it cited the depths' printed page number (20), not the PDF page (14).",
        "fix": "Chunks find the page, then the whole page is read. A prompt rewording stops tender questions being refused, and every p.N citation is re-checked against the cited page's numbers and corrected.",
    },
    "q15": {
        "baseline": "wrong", "final": "correct",
        "diagnosis": "Called HillOfTowie_18 'closest to the Betz limit' with Cp 0.5995, which is above it: a physically impossible value presented as a success. Later runs named the second-highest turbine from the unsorted list.",
        "fix": "Tools now flag a peak Cp above 0.593 as a measurement artifact (nacelle anemometer bias), and the analysis tool returns its extremes ready-made.",
    },

    # ---- set 2: new questions, asked after set 1's fixes ----
    "q16": {"baseline": "correct", "final": "correct", "diagnosis": "Best day 2024-02-02, 1,057.9 MWh, from farm facts.", "fix": None},
    "q17": {
        "baseline": "wrong", "final": "correct",
        "diagnosis": "Named 2020 (80.79 GWh). The best full year is 2017 (84.83 GWh); the agent misread the year list.",
        "fix": "Farm facts now give the best full calendar year ready-made (years under 95% SCADA coverage excluded).",
    },
    "q18": {"baseline": "correct", "final": "correct", "diagnosis": "Kelmarsh 3 and 6 at 68.5 m, the rest at 78.5 m.", "fix": None},
    "q19": {
        "baseline": "refused", "final": "correct",
        "diagnosis": "Sent to the fault-log RAG, which has nothing on AeroUp, then refused. The dates were in farm facts.",
        "fix": "The routing prompt names upgrades (AeroUp) as a farm-facts question.",
    },
    "q20": {"baseline": "correct", "final": "correct", "diagnosis": "0.859% mean electrical loss, from OpenOA.", "fix": None},
    "q21": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Right turbine (Penmanshiel_10, 18.82%), but printed the prompt's own citation template, '(document title, p.N)'. "
                     "A later run read the unsorted OpenOA wake list instead and named the second-highest turbine.",
        "fix": "Template text removed from the prompt and stripped from answers; the OpenOA summary names the largest turbine wake loss.",
    },
    "q22": {
        "baseline": "wrong", "final": "correct",
        "diagnosis": "Invented a P50 of 53.385 GWh: that number is the turbines' metered energy from the electrical-loss study. "
                     "No AEP was computed for this farm (one year, no availability channel).",
        "fix": "The summary now says in so many words that no P50/P90 exists for this farm, and why.",
    },
    "q23": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Right total (1.169%), wrong largest component: said windiness (0.499%), it's the regression (0.702%).",
        "fix": "The summary names the largest uncertainty component ready-made.",
    },
    "q24": {"baseline": "correct", "final": "correct", "diagnosis": "6-18 rpm and 3-5 m/s, cited p.6.", "fix": None},
    "q25": {"baseline": "correct", "final": "correct", "diagnosis": "Asynchronous generator, integrated heat exchanger (p.6).", "fix": None},
    "q26": {"baseline": "correct", "final": "correct", "diagnosis": "2.125 MW for 60 Hz, not the 50 Hz 2.2 MW.", "fix": None},
    "q27": {"baseline": "correct", "final": "correct", "diagnosis": "The bat curtailment rules, cited p.33.", "fix": None},
    "q28": {"baseline": "correct", "final": "correct", "diagnosis": "7.0 m and 40.4 m/s, p.23. The automatic grade failed it on 'meters per second': a grader error.", "fix": "Grader pattern corrected."},
    "q29": {
        "baseline": "refused", "final": "correct",
        "diagnosis": "Answered 'I can only provide information about Penmanshiel' without calling a tool: the farm-scoped prompt read as a ban on tender questions.",
        "fix": "The system prompt says tender and leasing questions are in scope whichever farm is selected.",
    },
    "q30": {"baseline": "correct", "final": "correct", "diagnosis": "Kelmarsh_2, 0.2999.", "fix": None},
    "q31": {"baseline": "correct", "final": "correct", "diagnosis": "Penmanshiel_01, 37 maintenance stops.", "fix": None},
    "q32": {
        "baseline": "correct", "final": "correct",
        "diagnosis": "Correctly said forced outages aren't identifiable here, but only after three detail calls: the empty ranking carried no reason.",
        "fix": "The empty ranking now says why it's empty.",
    },
    "q33": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Didn't invent anything, but said 'no information' instead of pointing out Kelmarsh 7 doesn't exist.",
        "fix": "The system prompt lists the farm's turbines and says a turbine outside the list doesn't exist.",
    },
    "q34": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Reported the 50.4% mean hourly error as '0.504%' (a fraction under a 'pct' name). The automatic grade passed it. "
                     "After a prompt note failed, a run also called 5.654 MW actual vs 5.653 predicted 'slightly lower'.",
        "fix": "The tool now returns the error as a percentage and the direction in words; the grader forbids both mistakes.",
    },
    "q35": {"baseline": "correct", "final": "correct", "diagnosis": "HillOfTowie_18, +0.31 m/s power-curve displacement. HillOfTowie_16 was also flagged before the rotor fix.", "fix": None},
}
