"""
=============================================================
  Pocket Legal Assistant — Streamlit Application
  app.py
=============================================================
  UI: Professional, minimalist "legal document" aesthetic.
      No chat history. Single input -> single output.
  ML: Loads legal_model.pkl + vectorizer.pkl from disk.
=============================================================
"""

import streamlit as st
import joblib
import os

# ─────────────────────────────────────────────────────────────
# RESPONSE DICTIONARY
# Each entry maps a Legal_Category to a structured legal response.
# Includes: disclaimer, relevant law, and a practical next step.
# ─────────────────────────────────────────────────────────────
LEGAL_RESPONSES = {
    "Traffic_Harassment": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Traffic Harassment / Police Misconduct**

**Relevant Law**
- Motor Vehicles Act, 1988 — governs rash driving, road rage, and conduct of traffic enforcement officers.
- Indian Penal Code, Section 294 — obscene acts or words in a public place.
- Indian Penal Code, Section 506 — criminal intimidation.
- Prevention of Corruption Act, 1988 — applies when an officer demands an illegal gratification (bribe).

**Suggested Next Steps**
1. Note the officer's name, badge number, vehicle number of any involved party, exact time, and location.
2. If keys were wrongfully seized or a bribe was demanded, file a written complaint with the Senior Police Inspector or Deputy Commissioner of Police (Traffic) of your area.
3. You may also lodge a complaint with the State Anti-Corruption Bureau or call the Police Vigilance helpline.
4. For general traffic violations by civilians, file an FIR at your nearest police station.
5. If you feel physically unsafe, contact the Police Control Room immediately.

**Note:** Retain any dashcam footage, witness contact details, or photographs. These significantly strengthen your complaint.
""",

    "Tenant_Rights": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Tenant Rights Violation**

**Relevant Law**
- The Rent Control Act (state-specific — e.g., Delhi Rent Control Act, Maharashtra Rent Control Act, 1999).
- Transfer of Property Act, 1882 — governs the rights and obligations under a tenancy agreement.
- Indian Penal Code, Section 441 — criminal trespass, applicable if the landlord enters without consent.

**Suggested Next Steps**
1. Do not vacate the premises without a formal court eviction order. Verbal demands carry no legal force.
2. Send a legal notice via registered post to the landlord documenting all grievances — illegal entry, withheld deposit, utility disconnection, or harassment.
3. File a complaint before the Rent Control Tribunal or Rent Court in your jurisdiction.
4. Preserve all evidence: rent receipts, bank transfer records, photographs, and all written communication.

**Note:** A registered rent agreement is your strongest evidentiary document. Always insist on one.
""",

    "Cybercrime_Financial_Fraud": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Cybercrime — Financial Fraud**

**Relevant Law**
- Information Technology Act, 2000 — Section 66C (identity theft), Section 66D (cheating by impersonation using a computer resource).
- Indian Penal Code, Section 420 — cheating and dishonestly inducing delivery of property.
- Indian Penal Code, Section 468 — forgery for the purpose of cheating.

**Suggested Next Steps**
1. Act immediately. Contact your bank's 24-hour fraud helpline to block your account or card and dispute the transaction.
2. Report online at the National Cyber Crime Reporting Portal: https://cybercrime.gov.in or call the Cyber Crime Helpline: 1930.
3. File an FIR at your local police station or the nearest Cyber Crime Cell, providing all transaction records, screenshots, and call logs.
4. Request your bank to initiate a chargeback or transaction reversal without delay.

**Note:** Never share OTPs, CVVs, or PINs with any caller, regardless of the organisation they claim to represent.
""",

    "Cybercrime_Harassment": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Cybercrime — Online Harassment**

**Relevant Law**
- Information Technology Act, 2000 — Section 67 (obscene material online), Section 66E (violation of privacy), Section 66C/66D (identity impersonation).
- Indian Penal Code, Section 354D — stalking, including cyber-stalking.
- Indian Penal Code, Sections 499 and 500 — defamation.
- Indian Penal Code, Section 384 — extortion or blackmail.

**Suggested Next Steps**
1. Do not comply with any demands if you are being blackmailed. Payment invites further extortion.
2. Preserve all evidence: screenshots of messages, profile URLs, and timestamps before blocking the offender.
3. Use the platform's built-in reporting tools (Instagram, Facebook, WhatsApp) to report and block the account.
4. File a complaint at https://cybercrime.gov.in or call 1930.
5. File an FIR at the Cyber Crime Cell of your local police station.

**Note:** For non-consensual image-based abuse, the Internet Crimes Against Children unit can also be engaged.
""",

    "Consumer_Protection": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Consumer Rights Violation**

**Relevant Law**
- Consumer Protection Act, 2019 — covers defective goods, deficient services, unfair trade practices, and misleading advertisements.
- Food Safety and Standards Act, 2006 — applicable to food quality or adulteration complaints.
- Legal Metrology Act, 2009 — governs overcharging and hidden fees.

**Suggested Next Steps**
1. Send a formal legal notice to the business or seller, documenting the issue and demanding resolution within 15 days.
2. File a complaint on the National Consumer Helpline: 1800-11-4000 (toll-free) or at https://consumerhelpline.gov.in.
3. Escalate to the District Consumer Disputes Redressal Commission (DCDRC) for disputes up to Rs. 50 lakhs. The process is straightforward and filing fees are minimal.
4. Retain all receipts, warranty documentation, screenshots, and correspondence with the company.

**Note:** Under the Consumer Protection Act, 2019, you are entitled to a refund, repair, replacement, or compensation for defective goods and deficient services.
""",

    "Employment_Dispute": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Employment or Labour Dispute**

**Relevant Law**
- Industrial Disputes Act, 1947 — wrongful termination, lay-offs, retrenchment.
- Payment of Wages Act, 1936 — unpaid or delayed salary.
- Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 (POSH Act).
- Maternity Benefit Act, 1961 — maternity leave entitlements.
- Employees' Provident Funds and Miscellaneous Provisions Act, 1952 — PF non-deposit.
- Indian Penal Code, Section 383 — extortion, applicable to forced resignations under duress.

**Suggested Next Steps**
1. Secure all documentation: pay slips, employment contract, offer letter, emails, and any written communication.
2. Send a formal legal notice to the HR department and management through a registered advocate.
3. File a complaint with the Labour Commissioner of your state.
4. For POSH violations, approach the Internal Complaints Committee (ICC) of your company or the Local Complaints Committee (LCC) if the company employs fewer than 10 persons.
5. For PF-related issues, raise a grievance at the EPFO portal: https://epfigms.gov.in or call 1800-118-005.

**Note:** A forced resignation under duress may be treated as constructive dismissal in law. Consult a labour lawyer immediately.
""",

    "Property_Dispute": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Property Dispute**

**Relevant Law**
- Transfer of Property Act, 1882 — governs sale, gift, lease, and mortgage of immovable property.
- Hindu Succession Act, 1956 / Indian Succession Act, 1925 — inheritance and division of ancestral property.
- Indian Penal Code, Section 447 — criminal trespass.
- Indian Penal Code, Sections 467 and 468 — forgery of property documents and deeds.
- Specific Relief Act, 1963 — court-ordered possession or specific performance of a contract.

**Suggested Next Steps**
1. Secure all original documents immediately: sale deed, title deed, gift deed, mutation records, and survey records.
2. File for a stay order in civil court to prevent further encroachment, unauthorised transfer, or demolition.
3. File an FIR if criminal trespass, fraud, or document forgery is involved.
4. File a civil suit for declaration of title, recovery of possession, or partition before the competent civil court.
5. For builder-related disputes or delayed possession, file a complaint with the Real Estate Regulatory Authority (RERA) of your state.

**Note:** A registered sale deed combined with documented, continuous peaceful possession is the strongest proof of ownership in Indian courts.
""",

    "Public_Nuisance": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Public Nuisance**

**Relevant Law**
- Indian Penal Code, Section 268 — definition of a public nuisance.
- Indian Penal Code, Section 290 — punishment for committing a public nuisance.
- Code of Criminal Procedure, Section 133 — empowers a Magistrate to issue a conditional order for removal of a public nuisance.
- Noise Pollution (Regulation and Control) Rules, 2000 — applicable to noise-related complaints.
- Environment (Protection) Act, 1986 — applicable to pollution and environmental hazards.

**Suggested Next Steps**
1. File a written complaint with your local police station or the Executive Magistrate of your area.
2. For noise violations, file a complaint with the State Pollution Control Board.
3. Report infrastructure issues such as open drains, uncollected garbage, or stray animals to the Municipal Corporation or Gram Panchayat via their civic helpline.
4. Any citizen can file a Section 133 CrPC petition before the District Magistrate to compel removal of the nuisance.

**Note:** A collective complaint signed by multiple residents carries considerably more weight with authorities and local administration.
""",

    "Theft": """
**Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**Category Identified: Theft**

**Relevant Law**
- Indian Penal Code, Section 378 — definition of theft.
- Indian Penal Code, Section 379 — punishment for theft (up to 3 years imprisonment, fine, or both).
- Indian Penal Code, Section 356 — assault or criminal force in committing theft (snatching).
- Indian Penal Code, Sections 392 and 394 — robbery and grievous hurt in the commission of robbery.
- Indian Penal Code, Section 380 — theft committed in a dwelling house.

**Suggested Next Steps**
1. File an FIR immediately at the nearest police station. The FIR number is essential for insurance claims, bank fraud reports, and subsequent legal proceedings.
2. If a vehicle was stolen, also notify the Regional Transport Office (RTO) and your insurance provider within 24 hours of the incident.
3. For mobile phone theft, report the IMEI number to your telecom carrier and register it at https://ceir.gov.in to initiate a block.
4. For suspected domestic theft by household staff, inform the police and provide any available evidence.

**Note:** Filing an FIR for even minor theft creates an official record, which is valuable if the offender is apprehended at a later stage.
""",
}


# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pocket Legal Assistant",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — clean legal-document aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      color: #1a1a1a;
  }

  .stApp {
      background-color: #f7f5f0;
  }

  /* ── Main content block ── */
  .block-container {
      max-width: 760px !important;
      padding-top: 3.5rem !important;
      padding-bottom: 4rem !important;
  }

  /* ── Masthead ── */
  .masthead {
      border-bottom: 2px solid #1a1a1a;
      padding-bottom: 1.2rem;
      margin-bottom: 0.4rem;
  }
  .masthead-kicker {
      font-family: 'Inter', sans-serif;
      font-size: 0.68rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #666;
      margin-bottom: 0.5rem;
  }
  .masthead-title {
      font-family: 'EB Garamond', Georgia, serif;
      font-size: 2.6rem;
      font-weight: 600;
      color: #0d0d0d;
      line-height: 1.15;
      margin: 0;
  }
  .masthead-sub {
      font-family: 'Inter', sans-serif;
      font-size: 0.85rem;
      color: #555;
      margin-top: 0.55rem;
      line-height: 1.6;
  }

  /* ── Divider ── */
  .rule {
      border: none;
      border-top: 1px solid #c8c3ba;
      margin: 1.6rem 0;
  }

  /* ── Section label ── */
  .section-label {
      font-size: 0.68rem;
      font-weight: 600;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #888;
      margin-bottom: 0.5rem;
  }

  /* ── Text area ── */
  .stTextArea textarea {
      background-color: #ffffff !important;
      border: 1px solid #b0a99e !important;
      border-radius: 0 !important;
      color: #1a1a1a !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.93rem !important;
      line-height: 1.65 !important;
      padding: 0.85rem 1rem !important;
      box-shadow: none !important;
      transition: border-color 0.15s ease;
  }
  .stTextArea textarea:focus {
      border-color: #1a1a1a !important;
      box-shadow: none !important;
      outline: none !important;
  }
  .stTextArea textarea::placeholder {
      color: #aaa !important;
      font-style: italic;
  }
  .stTextArea label {
      display: none !important;
  }

  /* ── Submit button ── */
  .stButton > button {
      background-color: #0f1923 !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 0 !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
      padding: 0.7rem 2.2rem !important;
      transition: background-color 0.15s ease !important;
      cursor: pointer;
      width: auto !important;
  }
  .stButton > button:hover {
      background-color: #1e3a5f !important;
  }
  .stButton > button:active {
      background-color: #0a0f15 !important;
  }

  /* ── Result panel ── */
  .result-panel {
      border-left: 3px solid #0f1923;
      background-color: #ffffff;
      padding: 1.8rem 2rem;
      margin-top: 2rem;
  }
  .result-category-label {
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #888;
      margin-bottom: 0.3rem;
  }
  .result-category-value {
      font-family: 'EB Garamond', Georgia, serif;
      font-size: 1.45rem;
      font-weight: 600;
      color: #0f1923;
      margin-bottom: 1.2rem;
      border-bottom: 1px solid #e0dbd3;
      padding-bottom: 0.8rem;
  }
  .result-body {
      font-family: 'Inter', sans-serif;
      font-size: 0.88rem;
      color: #2a2a2a;
      line-height: 1.78;
  }
  .result-body strong {
      color: #0f0f0f;
      font-weight: 600;
  }
  .result-body hr {
      border: none;
      border-top: 1px solid #e0dbd3;
      margin: 1rem 0;
  }

  /* ── Warning / error ── */
  .stAlert {
      border-radius: 0 !important;
      font-size: 0.88rem;
  }

  /* ── Footer ── */
  .footer-line {
      margin-top: 3rem;
      border-top: 1px solid #c8c3ba;
      padding-top: 1rem;
      font-size: 0.72rem;
      color: #999;
      letter-spacing: 0.04em;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL & VECTORISER (cached for speed)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path      = "legal_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    return joblib.load(model_path), joblib.load(vectorizer_path)

model, vectorizer = load_artifacts()


# ─────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="masthead-kicker">Academic Project &nbsp;&mdash;&nbsp; AI-Assisted Legal Guidance</div>
  <div class="masthead-title">Pocket Legal Assistant</div>
  <div class="masthead-sub">
    Describe your legal situation in plain language. The system will classify your case
    and return the relevant statute and suggested course of action.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL STATUS CHECK
# ─────────────────────────────────────────────
if model is None or vectorizer is None:
    st.error(
        "Model files not found. Please run the training script first:\n\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Describe your situation</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="scenario",
    placeholder="e.g. My landlord is refusing to return my security deposit and entered my flat without permission...",
    height=150,
    max_chars=2000,
)

submitted = st.button("Analyse")

# ─────────────────────────────────────────────
# PREDICTION & OUTPUT
# ─────────────────────────────────────────────
if submitted:
    text = user_input.strip()

    if not text:
        st.warning("Please describe your situation before submitting.")
    else:
        input_vec          = vectorizer.transform([text])
        predicted_category = model.predict(input_vec)[0]
        legal_response     = LEGAL_RESPONSES.get(
            predicted_category,
            f"Category identified: **{predicted_category}**\n\n"
            "Please consult a qualified legal professional for personalised advice."
        )

        # Display category label in a clean panel via HTML wrapper,
        # then render the markdown response body natively.
        display_name = predicted_category.replace("_", " ").title()

        st.markdown(f"""
        <div class="result-panel">
          <div class="result-category-label">Legal Classification</div>
          <div class="result-category-value">{display_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Render the rich markdown response inside Streamlit natively
        # (so bold, links, and lists render correctly)
        st.markdown(
            "<div class='result-body'></div>",
            unsafe_allow_html=True
        )
        st.markdown(legal_response)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-line">
  Pocket Legal Assistant &nbsp;&mdash;&nbsp; Academic Project &nbsp;&mdash;&nbsp;
  This tool does not provide professional legal advice. Always consult a qualified advocate.
</div>
""", unsafe_allow_html=True)
