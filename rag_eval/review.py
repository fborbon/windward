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
    "q11": {"baseline": "correct", "final": "correct", "diagnosis": "82.4 m datasheet vs 82.0 m configured, from the verified cross-check.", "fix": None},
    "q12": {
        "baseline": "partial", "final": "partial",
        "diagnosis": "Right upgrade, 2.2 MW and up to 5%, but uncited, and it drops the sheet's scope note (only turbines with a Bachmann controller).",
        "fix": "Citations now kept. The scope note is still dropped despite an explicit prompt rule: a remaining Nova Lite failure.",
    },
    "q13": {
        "baseline": "partial", "final": "correct",
        "diagnosis": "Right six areas, but leaked a '</thinking>' tag and a sentence of planning. Later runs also padded each item with invented explanations, or returned an empty answer.",
        "fix": "Reasoning tags stripped, a rule against explanations the page doesn't give, and one retry on an empty reply. "
               "The final answer cites the whole document (p.1-5) where p.2 and p.4 would be exact.",
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
}
