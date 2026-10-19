"""
make_seed_data.py -> data/extra_seed.tsv
A synthetic SEED set of modern Indian-style SMS: legit transactional/personal (ham) vs
common scam patterns (spam). It teaches the models what the 2005 SMS corpus never showed
(OTPs, bank alerts, delivery updates vs KYC/electricity/job/parcel scams).
It is a starting point, NOT a substitute for real messages: add real ones to
data/extra_<anything>.tsv (label<TAB>text) or just use the feedback buttons.
"""
import random
random.seed(7)
P = lambda xs: random.choice(xs)
n = lambda a, b: random.randint(a, b)

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak", "Canara Bank", "PNB", "Bank of Baroda"]
MERCH = ["Zomato", "Swiggy", "Amazon", "Flipkart", "BigBasket", "IRCTC", "Myntra", "Uber", "Ola", "Blinkit", "Jio", "Airtel"]
DOMS = ["swiggy.com", "amazon.in", "flipkart.com", "zomato.com", "myntra.com", "irctc.co.in", "bluedart.com", "delhivery.com"]
TELCO = ["Jio", "Airtel", "Vi", "BSNL"]
COUR = ["FedEx", "DHL", "India Post", "BlueDart", "Delhivery", "Amazon"]
TLDS = ["xyz", "top", "tk", "click", "icu", "buzz", "live", "online", "site", "vip"]
KW = ["kyc", "update", "verify", "secure", "login", "claim", "refund", "pan", "reward", "support"]
def date(): return f"{n(1,28):02d}-{P(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}"
def amt(): return f"{n(10,99999):,}.{n(0,99):02d}"
def phone(): return P(["", "+91"]) + str(n(6,9)) + "".join(str(n(0,9)) for _ in range(9))
def bad_url():
    b = P(["sbi", "hdfc", "icici", "paytm", "amazon", "axis", "jio", "airtel", "aadhaar", "upi"])
    return P([f"{b}-{P(KW)}.{P(TLDS)}/{P(KW)}", f"{P(KW)}-{b}-{P(KW)}.{P(TLDS)}", f"bit.ly/{''.join(random.choices('abcXYZ0123456789',k=7))}",
              f"http://{b}{P(KW)}.{P(TLDS)}/login", f"tinyurl.com/{''.join(random.choices('abcdefg23456',k=6))}"])

HAM = [
 lambda: f"{n(100000,999999)} is your OTP for {P(MERCH)}. Do not share it with anyone.",
 lambda: f"Your OTP for login is {n(100000,999999)}. Valid for 10 minutes. Never share this code with anyone.",
 lambda: f"OTP {n(1000,999999)} for transaction of Rs {amt()} at {P(MERCH)}. Do not share. -{P(BANKS)}",
 lambda: f"Rs {amt()} debited from A/c XX{n(1000,9999)} on {date()} at {P(MERCH)}. Avl bal Rs {amt()}. -{P(BANKS)}",
 lambda: f"Rs.{amt()} credited to your A/c XX{n(1000,9999)} on {date()} by UPI Ref {n(10**11,10**12-1)}. -{P(BANKS)}",
 lambda: f"UPI txn of Rs {amt()} to {P(MERCH)} is successful. Ref no {n(10**11,10**12-1)}.",
 lambda: f"Your {P(MERCH)} order #{n(10**6,10**7)} is out for delivery today. Track: https://www.{P(DOMS)}/track/{n(10**5,10**6)}",
 lambda: f"Your {P(MERCH)} order has been shipped and will arrive by {date()}. Thank you for shopping with us.",
 lambda: f"Your {P(['Uber','Ola'])} ride is arriving. Driver {P(['Ravi','Suresh','Kumar','Anita'])}, TN{n(10,99)}AB{n(1000,9999)}. Share OTP {n(1000,9999)} to start.",
 lambda: f"Reminder: your appointment with Dr {P(['Sharma','Rao','Iyer','Khan'])} is on {date()} at {n(1,12)}:{P(['00','30'])} {P(['AM','PM'])}.",
 lambda: f"Your {P(BANKS)} credit card statement for {P(['Aug','Sep','Oct'])} is generated. Total due Rs {amt()}, due date {date()}. Pay via the official app.",
 lambda: f"Dear customer, your {P(TELCO)} recharge of Rs {n(99,999)} is successful. Validity till {date()}.",
 lambda: f"Meeting moved to {n(1,6)}pm. Join: https://meet.google.com/abc-defg-hij",
 lambda: f"Class on {P(['Control Systems','Verilog lab','DSP','Probability','Signals'])} shifted to room {P(['AB1','SJT','TT','MB'])}-{n(100,699)} at {n(8,11)}:{P(['00','30'])}.",
 lambda: f"Dont forget the {P(['assignment','lab record','project review','CAT'])} submission on vtop tomorrow",
 lambda: P(["bro are we still meeting at the canteen at 5", "did you finish the verilog assignment yet", "call me when you reach home",
            "kal lab hai kya? bring the record", "ok see you in 10 mins", "can you send the notes for control systems please",
            "happy birthday! party at my place on saturday", "lunch at 1? i'm at the library", "mom said dinner is ready come home",
            "thanks da, will pay you back tomorrow", "movie tonight? i booked 2 tickets on bookmyshow", "reached the hostel, call you later"]),
 lambda: f"Your EMI of Rs {amt()} for {P(BANKS)} loan will be auto-debited on {date()}. Please maintain sufficient balance.",
 lambda: f"IRCTC: PNR {n(10**9,10**10-1)} confirmed. Train {n(10000,29999)} departs {date()}. Wish you a happy journey.",
 lambda: f"Your electricity bill of Rs {amt()} for the month is generated. Pay by {date()} on the official TNEB portal or app.",
 lambda: f"Rs {amt()} spent on {P(BANKS)} Card XX{n(1000,9999)} at {P(MERCH)} on {date()}. Not you? Call the number on the back of your card.",
 lambda: f"Your OTP to reset {P(['VTOP','Gmail','Netflix','Paytm','Instagram'])} password is {n(100000,999999)}. It expires in {n(3,10)} minutes.",
 lambda: f"Hi, your {P(TELCO)} bill for {P(['Aug','Sep','Oct'])} is Rs {n(199,1299)}. Pay anytime in the {P(TELCO)} app.",
 lambda: f"Alert: {P(BANKS)} will never ask for your PIN, OTP or password. Do not share them with anyone, including bank staff.",
]
SPAM = [
 lambda: f"Dear customer your {P(BANKS)} account will be blocked today. Update KYC now {bad_url()}",
 lambda: f"{P(BANKS)} ALERT: your PAN card is not updated. Account will be suspended. Click {bad_url()} immediately",
 lambda: f"Your electricity power will be disconnected tonight 9:30PM due to unpaid bill. Call {phone()} immediately to update",
 lambda: f"Earn Rs {n(2,9)*1000} daily working from home part time. Message on WhatsApp {phone()} now",
 lambda: f"Congratulations! You have won Rs {n(1,50)},00,000 in {P(['KBC','Amazon','Jio','Flipkart'])} lucky draw. Claim now: {bad_url()}",
 lambda: f"{P(COUR)}: your parcel is held at customs. Pay Rs {n(20,99)} redelivery fee here {bad_url()}",
 lambda: f"Pre-approved loan of Rs {n(1,5)},00,000 no documents instant approval in 5 minutes. Apply {bad_url()}",
 lambda: f"Your {P(BANKS)} credit card reward points worth Rs {amt()} expire today. Redeem now {bad_url()}",
 lambda: f"Refund of Rs {amt()} is pending in your account. Click {bad_url()} to claim before it expires",
 lambda: f"Your SIM will be disconnected in 2 hours as per TRAI rules. Complete verification call {phone()} now",
 lambda: f"Join our VIP trading group and get {n(2,9)}00% guaranteed returns daily. WhatsApp {phone()} limited seats",
 lambda: f"Your Aadhaar is linked to illegal activity. Cyber cell notice. Call {phone()} immediately to avoid arrest",
 lambda: f"URGENT: unusual login detected on your {P(MERCH)} account. Verify your identity now {bad_url()} or account will be locked",
 lambda: f"Dear user your {P(BANKS)} netbanking is deactivated. Re-activate here {bad_url()} Enter card number and PIN",
 lambda: f"Free recharge Rs {n(199,999)} for all {P(TELCO)} users! Click {bad_url()} to get your free gift",
 lambda: f"You are selected for work from home job. Salary Rs {n(30,90)},000 per month. Registration fee Rs {n(99,999)} only. Contact {phone()}",
 lambda: f"Last warning: pay Rs {n(500,5000)} now or your private photos will be shared. Pay at {bad_url()}",
 lambda: f"Bonus offer! Deposit Rs {n(100,500)} and get Rs {n(1000,5000)} free in online betting. Play {bad_url()}",
]
rows = set()
for _ in range(900):
    rows.add("ham\t" + random.choice(HAM)())
    rows.add("spam\t" + random.choice(SPAM)())
import os; os.makedirs("data", exist_ok=True)
open("data/extra_seed.tsv", "w", encoding="utf-8").write("\n".join(sorted(rows)) + "\n")
print(len(rows), "raw rows written (duplicates after masking are merged by train_all)")
