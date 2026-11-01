"""Message-pattern scam check (one low-effort structured call)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from config import agent_model, reasoning_kwargs

SIGNALS = (
    "urgency",
    "prize_lottery",
    "kyc_block",
    "otp_pin",
    "job_pay",
    "fake_bill_refund",
    "install_app",
    "payment",
)

INSTRUCTIONS = (
    "You flag scam message patterns in a forwarded message. "
    "Return signals ONLY from this list, in any order: urgency "
    "(deadline/account-block pressure), prize_lottery (won a prize, lottery, "
    "cashback requiring action), kyc_block (KYC update or account will be "
    "blocked/suspended), otp_pin (asks for OTP, PIN, CVV, or password), "
    "job_pay (part-time job, paid tasks, rating/like jobs), "
    "fake_bill_refund (fake electricity/water bill, refund, or recharge "
    "offer), install_app (tells the reader to install an app or APK), "
    "payment (asks to pay, send money, or share a UPI ID/QR). "
    "scam_type is the scam in plain words in the message's language "
    "(e.g. 'Fake KYC update', 'Fake electricity bill'); use '' when no "
    "signal applies. Never invent signals outside the list."
)


class ScamReading(BaseModel):
    scam_type: str = Field(description="Scam in plain words, or ''.")
    signals: list[str] = Field(description="Subset of the fixed vocabulary.")


def read_scam_text(
    text: str, language: str, *, client, tracker=None
) -> ScamReading:
    """Classify scam patterns; unknown signal names are dropped."""
    if tracker is not None:
        tracker.mark("scamtext")
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=f"Language: {language}\n\n{text}",
        text_format=ScamReading,
        **reasoning_kwargs("scamtext"),
    )
    if tracker is not None:
        tracker.add(response.usage, stage="scamtext")
    reading = response.output_parsed
    reading.signals = [s for s in reading.signals if s in SIGNALS]
    return reading
