"""
Risk scoring (feature 1) and 3-tier classification (feature 3).

Instead of a raw binary SPAM/HAM label, we combine both models' probabilities
into a single 0-100 "risk score" and map it onto three tiers: SAFE,
SUSPICIOUS, and HIGH RISK. The middle "suspicious" tier matters because a lot
of real fraud messages sit in an ambiguous zone — not obviously ham, not
overwhelmingly spam either — and a binary classifier forces a decision where
none is confidently justified. Surfacing that ambiguity is more honest than
hiding it behind a coin-flip threshold.
"""

# Thresholds are configurable so you can justify/tune them in your report.
SAFE_MAX = 30          # risk_score below this -> SAFE
SUSPICIOUS_MAX = 70    # risk_score below this (and >= SAFE_MAX) -> SUSPICIOUS
                        # risk_score >= this -> HIGH RISK


def compute_risk_score(nb_prob, lstm_prob, nb_weight=0.5, lstm_weight=0.5):
    """
    Combine both models' spam probabilities into a single 0-100 risk score.
    Default is a simple weighted average; weights can be tuned if one model
    is trusted more (e.g. give the LSTM slightly more weight since it has
    higher precision in our evaluation).
    """
    combined = nb_weight * nb_prob + lstm_weight * lstm_prob
    return round(combined * 100, 1)


def risk_level(risk_score):
    """Map a 0-100 risk score onto a 3-tier label."""
    if risk_score < SAFE_MAX:
        return "SAFE"
    elif risk_score < SUSPICIOUS_MAX:
        return "SUSPICIOUS"
    else:
        return "HIGH RISK"


def risk_summary(nb_prob, lstm_prob, nb_weight=0.5, lstm_weight=0.5):
    """Convenience wrapper returning both the score and the tier together."""
    score = compute_risk_score(nb_prob, lstm_prob, nb_weight, lstm_weight)
    return {"risk_score": score, "level": risk_level(score)}


if __name__ == "__main__":
    # Quick sanity check across a spread of example probability pairs
    examples = [
        (0.01, 0.02),   # clearly safe
        (0.4, 0.35),    # ambiguous -> should land in SUSPICIOUS
        (0.55, 0.6),    # ambiguous, leaning spam -> SUSPICIOUS
        (0.95, 0.99),   # clearly spam -> HIGH RISK
    ]
    for nb, lstm in examples:
        summary = risk_summary(nb, lstm)
        print(f"NB={nb:.2f} LSTM={lstm:.2f} -> risk={summary['risk_score']}  level={summary['level']}")
