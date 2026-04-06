"""
=============================================================
  Pocket Legal Assistant — Streamlit Chatbot Interface
  app.py
=============================================================
  Workflow:
    1. Load the pre-trained model (legal_model.pkl) and
       TF-IDF vectoriser (vectorizer.pkl) from disk.
    2. Present a clean, chat-style UI to the user.
    3. Accept a free-text scenario from the user.
    4. Vectorise → Predict → Look-up response → Display.
=============================================================
"""

import streamlit as st
import joblib
import os
import time

# ─────────────────────────────────────────────────────────────
# RESPONSE DICTIONARY
# Maps each Legal_Category to a structured, helpful response.
# Each response includes:
#   • A disclaimer (not professional legal advice)
#   • The relevant general law / act
#   • A practical next step
# ─────────────────────────────────────────────────────────────
LEGAL_RESPONSES = {
    "Traffic_Harassment": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Traffic Harassment**

**Relevant Law:**
- *Motor Vehicles Act, 1988* — covers rash/negligent driving, road rage, and intimidating behaviour on public roads.
- *Indian Penal Code (IPC) Section 294* — obscene acts or words in a public place.
- *IPC Section 506* — criminal intimidation.

**What You Can Do:**
1. Note the vehicle number, time, location, and description of the incident.
2. **File an FIR** at your nearest police station under the relevant sections.
3. You may also report traffic violations via the **National Highway Traffic Helpline: 1033** or your state's traffic police portal.
4. If you feel physically unsafe, call **Police: 100** or **Women's Helpline: 1091** immediately.

**💡 Tip:** Dashcam footage or a witness statement greatly strengthens your complaint.
""",

    "Tenant_Rights": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Tenant Rights Violation**

**Relevant Law:**
- *The Rent Control Act* (applicable state-specific variant, e.g., Delhi Rent Control Act, Maharashtra Rent Control Act).
- *Transfer of Property Act, 1882* — governs landlord-tenant agreements.
- *IPC Section 441* — criminal trespass (if landlord enters without consent).

**What You Can Do:**
1. **Do not vacate** without a proper court eviction order — verbal instructions or informal notices have no legal force.
2. **Send a legal notice** (via a lawyer or registered post) to your landlord documenting all grievances — illegal entry, withheld deposit, harassment, etc.
3. File a complaint with the **Rent Control Tribunal / Rent Court** in your city.
4. Preserve all evidence: rent receipts, WhatsApp messages, photographs of property damage.

**💡 Tip:** Always insist on a registered rent agreement — it is your strongest proof.
""",

    "Cybercrime_Financial_Fraud": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Cybercrime — Financial Fraud**

**Relevant Law:**
- *Information Technology Act, 2000* — Section 66C (identity theft), Section 66D (cheating by impersonation using computer resource).
- *Indian Penal Code (IPC) Section 420* — cheating and dishonestly inducing delivery of property.
- *IPC Section 468* — forgery for the purpose of cheating.

**What You Can Do:**
1. **Act immediately** — call your bank's 24/7 helpline to block your account/card and dispute the transaction. Act within the first hour if possible.
2. **Report online** at the National Cyber Crime Reporting Portal: **https://cybercrime.gov.in** or call **Cyber Crime Helpline: 1930**.
3. File an **FIR** at your local police station or the **Cyber Crime Cell** with all transaction details, screenshots, and call logs.
4. Freeze the fraudulent UPI/bank account by contacting your respective bank's fraud team.

**💡 Tip:** Never share OTPs, CVVs, or PINs with anyone, even someone claiming to be from your bank.
""",

    "Cybercrime_Harassment": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Cybercrime — Online Harassment**

**Relevant Law:**
- *Information Technology Act, 2000* — Section 67 (obscene material online), Section 66E (violation of privacy), Section 66C/66D (impersonation).
- *IPC Section 354D* — stalking (including cyber-stalking).
- *IPC Section 499 & 500* — defamation.
- *IPC Section 384* — extortion/blackmail.

**What You Can Do:**
1. **Do NOT comply** with blackmail demands — paying once only invites more demands.
2. Take **screenshots and preserve all evidence** (messages, profile URLs, timestamps).
3. **Report directly to the platform** (Instagram, Facebook, WhatsApp) using the 'Report' feature.
4. **File a complaint** at **https://cybercrime.gov.in** or call **1930**.
5. File an **FIR** at the Cyber Crime Cell of your local police station.

**💡 Tip:** For image-based abuse (non-consensual intimate images), the Internet Crimes Against Children (ICAC) unit can also be contacted.
""",

    "Consumer_Protection": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Consumer Rights Violation**

**Relevant Law:**
- *Consumer Protection Act, 2019* — covers defective goods, deficient services, unfair trade practices, and misleading advertisements.
- *Food Safety and Standards Act, 2006* — for food quality/adulteration complaints.
- *Legal Metrology Act, 2009* — for overcharging / hidden fees.

**What You Can Do:**
1. **Send a legal notice** to the business/seller documenting the issue and demanding remedy within 15 days.
2. **File a complaint** on the National Consumer Helpline: **1800-11-4000** (toll-free) or at **https://consumerhelpline.gov.in**.
3. Escalate to the **District Consumer Disputes Redressal Commission (DCDRC)** for disputes up to ₹50 lakhs — the process is simple and filing fees are minimal.
4. Keep all receipts, warranty cards, screenshots of ads, and communication with the company.

**💡 Tip:** You are entitled to a refund, repair, or replacement — and even compensation — for defective goods under the 2019 Act.
""",

    "Employment_Dispute": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Employment / Labour Dispute**

**Relevant Law:**
- *Industrial Disputes Act, 1947* — wrongful termination, lay-offs, retrenchment.
- *Payment of Wages Act, 1936* — unpaid / delayed salaries.
- *Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 (POSH Act)* — workplace sexual harassment.
- *Maternity Benefit Act, 1961* — maternity leave rights.
- *Employees' Provident Funds and Miscellaneous Provisions Act, 1952* — PF non-deposit.
- *IPC Section 383* — extortion (if forced to resign).

**What You Can Do:**
1. **Document everything**: save pay slips, employment contract, offer letter, emails, and any written communication.
2. **Send a legal notice** to HR and management via a registered lawyer.
3. File a complaint with the **Labour Commissioner** of your state.
4. For POSH violations, file with the **Internal Complaints Committee (ICC)** of your company or the **Local Complaints Committee (LCC)** if the company has fewer than 10 employees.
5. For PF issues, contact the **EPFO Grievance Portal: https://epfigms.gov.in** or call **1800-118-005**.

**💡 Tip:** Forced resignation under duress can be treated as constructive dismissal — consult a labour lawyer immediately.
""",

    "Property_Dispute": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Property Dispute**

**Relevant Law:**
- *Transfer of Property Act, 1882* — sale, gift, and mortgage of property.
- *Hindu Succession Act, 1956 / Indian Succession Act, 1925* — inheritance and ancestral property division.
- *IPC Section 420* — fraud / cheating (forged documents).
- *IPC Section 447* — criminal trespass.
- *IPC Section 467/468* — forgery of property documents / deeds.
- *Specific Relief Act, 1963* — court-ordered possession / specific performance of a contract.

**What You Can Do:**
1. **Secure all original documents**: sale deed, title deed, gift deed, mutation records, survey records.
2. File for a **stay order** in civil court immediately to prevent any further encroachment, transfer, or demolition.
3. File an **FIR** if there is criminal trespass, fraud, or forgery.
4. File a **civil suit** for declaration of title, possession, or partition in the competent civil court.
5. For builder-related delays or cheating, file a complaint with **RERA (Real Estate Regulatory Authority)** of your state.

**💡 Tip:** A registered sale deed combined with continuous, peaceful possession is the strongest proof of ownership in Indian courts.
""",

    "Public_Nuisance": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Public Nuisance**

**Relevant Law:**
- *IPC Section 268* — definition of public nuisance.
- *IPC Section 290* — punishment for public nuisance.
- *Code of Criminal Procedure (CrPC) Section 133* — Magistrate's order to remove public nuisance.
- *Noise Pollution (Regulation and Control) Rules, 2000* — for noise-related complaints.
- *Environment (Protection) Act, 1986* — for pollution and environmental hazards.

**What You Can Do:**
1. **File a written complaint** with your local police station or the **Executive Magistrate** of your area.
2. For noise violations, file a complaint with the **Pollution Control Board** of your state.
3. Complain to the **Municipal Corporation / Gram Panchayat** (for garbage, open drains, stray animals) via their helpline or civic portal.
4. You (or any community member) can file a **Section 133 CrPC petition** before the District Magistrate to direct the removal of the nuisance.

**💡 Tip:** A collective complaint signed by multiple residents carries significantly more weight with authorities.
""",

    "Theft": """
**⚠️ Disclaimer:** This information is for general awareness only and does not constitute professional legal advice. Please consult a qualified lawyer for advice specific to your situation.

---

**📋 Category Identified: Theft**

**Relevant Law:**
- *IPC Section 378* — definition of theft.
- *IPC Section 379* — punishment for theft (up to 3 years imprisonment / fine / both).
- *IPC Section 356* — assault or criminal force in committing theft (snatching).
- *IPC Section 392 / 394* — robbery and grievous hurt in robbery.
- *IPC Section 380* — theft in a dwelling house.

**What You Can Do:**
1. **File an FIR immediately** at your nearest police station — the FIR number is essential for insurance claims, bank fraud reports, and legal proceedings.
2. If a vehicle was stolen, also file a report with the **Regional Transport Office (RTO)** and your **insurance provider** within 24 hours.
3. For mobile phone theft, report the **IMEI number** to your carrier to block the device and at **https://ceir.gov.in**.
4. For suspected domestic theft (e.g., by household staff), share your suspicion with police — they can assist with verification.

**💡 Tip:** Even if the stolen item seems small, filing an FIR creates an official record that is useful if the thief is caught later.
""",
}

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pocket Legal Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — premium dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* ── Global background ── */
  .stApp {
      background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
      min-height: 100vh;
  }

  /* ── Header banner ── */
  .header-banner {
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.15);
      backdrop-filter: blur(12px);
      border-radius: 18px;
      padding: 2rem 2.5rem;
      margin-bottom: 1.8rem;
      text-align: center;
  }
  .header-banner h1 {
      color: #ffffff;
      font-size: 2.2rem;
      font-weight: 700;
      margin: 0 0 0.4rem 0;
      letter-spacing: -0.5px;
  }
  .header-banner p {
      color: rgba(255,255,255,0.65);
      font-size: 1rem;
      margin: 0;
  }
  .badge {
      display: inline-block;
      background: linear-gradient(90deg, #a78bfa, #60a5fa);
      border-radius: 999px;
      padding: 0.2rem 0.9rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: #fff;
      margin-bottom: 0.8rem;
      letter-spacing: 0.5px;
      text-transform: uppercase;
  }

  /* ── Chat bubble ── */
  .user-bubble {
      background: linear-gradient(135deg, #7c3aed, #4f46e5);
      border-radius: 18px 18px 4px 18px;
      padding: 1rem 1.3rem;
      margin: 0.5rem 0 1rem auto;
      max-width: 80%;
      color: #fff;
      font-size: 0.95rem;
      line-height: 1.6;
      box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
  }
  .bot-bubble {
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.13);
      backdrop-filter: blur(8px);
      border-radius: 18px 18px 18px 4px;
      padding: 1.2rem 1.5rem;
      margin: 0.5rem auto 1rem 0;
      max-width: 92%;
      color: #e2e8f0;
      font-size: 0.93rem;
      line-height: 1.75;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
  }
  .bot-bubble h2, .bot-bubble h3 {
      color: #a78bfa;
  }
  .bot-bubble strong {
      color: #f1f5f9;
  }
  .bot-bubble hr {
      border-color: rgba(255,255,255,0.12);
  }

  /* ── Label for tag ── */
  .category-tag {
      display: inline-block;
      background: linear-gradient(90deg, #10b981, #059669);
      color: #fff;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.25rem 0.8rem;
      border-radius: 999px;
      margin-bottom: 0.6rem;
      letter-spacing: 0.3px;
  }

  /* ── Input box ── */
  .stTextArea textarea {
      background: rgba(255,255,255,0.06) !important;
      border: 1.5px solid rgba(167, 139, 250, 0.4) !important;
      border-radius: 12px !important;
      color: #f1f5f9 !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.95rem !important;
      padding: 0.9rem !important;
      transition: border-color 0.2s ease;
  }
  .stTextArea textarea:focus {
      border-color: #a78bfa !important;
      box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
  }
  .stTextArea label {
      color: rgba(255,255,255,0.75) !important;
      font-weight: 500 !important;
  }

  /* ── Button ── */
  .stButton > button {
      background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 12px !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 1rem !important;
      font-weight: 600 !important;
      padding: 0.65rem 2rem !important;
      width: 100% !important;
      transition: opacity 0.2s ease, transform 0.15s ease !important;
      letter-spacing: 0.2px;
      cursor: pointer;
  }
  .stButton > button:hover {
      opacity: 0.88 !important;
      transform: translateY(-1px) !important;
  }
  .stButton > button:active {
      transform: translateY(0) !important;
  }

  /* ── Footer ── */
  .footer {
      text-align: center;
      color: rgba(255,255,255,0.3);
      font-size: 0.78rem;
      margin-top: 2.5rem;
      padding-bottom: 1rem;
  }

  /* ── Info box ── */
  .info-card {
      background: rgba(96, 165, 250, 0.1);
      border: 1px solid rgba(96, 165, 250, 0.3);
      border-radius: 12px;
      padding: 0.9rem 1.2rem;
      margin-bottom: 1.5rem;
      color: rgba(255,255,255,0.7);
      font-size: 0.88rem;
      line-height: 1.6;
  }

  /* ── Spinner text ── */
  .stSpinner > div {
      border-top-color: #a78bfa !important;
  }

  /* Hide Streamlit default elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & VECTORISER (cached for speed)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer():
    """Load the trained SVC model and TF-IDF vectoriser from disk."""
    model_path      = "legal_model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None

    model      = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ─────────────────────────────────────────────
# SESSION STATE (chat history)
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"/"bot", "text": ..., "category": ...}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="badge">⚖️ AI-Powered</div>
  <h1>Pocket Legal Assistant</h1>
  <p>Describe your situation in plain language. Get instant, categorised legal guidance.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL STATUS CHECK
# ─────────────────────────────────────────────
if model is None or vectorizer is None:
    st.error(
        "🚫 **Model files not found!**\n\n"
        "Please run the training script first:\n"
        "```\npython train_model.py\n```\n"
        "Then refresh this page."
    )
    st.stop()

# ─────────────────────────────────────────────
# INFO CARD
# ─────────────────────────────────────────────
st.markdown("""
<div class="info-card">
  💡 <strong>How to use:</strong> Describe your legal situation in your own words — the more detail, the better.
  The assistant will classify your case and provide relevant legal information and suggested next steps.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# USER INPUT FORM
# ─────────────────────────────────────────────
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area(
        label="✍️ Describe your situation",
        placeholder="e.g. My landlord is refusing to return my security deposit and is threatening to throw out my belongings...",
        height=130,
        max_chars=2000,
    )
    submitted = st.form_submit_button("🔍 Get Legal Guidance")

# ─────────────────────────────────────────────
# PREDICTION + RESPONSE LOGIC
# ─────────────────────────────────────────────
if submitted:
    user_input_stripped = user_input.strip()

    if not user_input_stripped:
        st.warning("⚠️ Please describe your situation before submitting.")
    else:
        # Append user turn to history
        st.session_state.chat_history.append({
            "role": "user",
            "text": user_input_stripped,
        })

        # Simulate a short bot "thinking" delay for UX polish
        with st.spinner("Analysing your situation …"):
            time.sleep(0.8)

            # 1. Vectorise the input
            input_vec = vectorizer.transform([user_input_stripped])

            # 2. Predict category
            predicted_category = model.predict(input_vec)[0]

            # 3. Fetch the canned response (fallback for unseen cats)
            bot_response = LEGAL_RESPONSES.get(
                predicted_category,
                f"**Legal Category: {predicted_category}**\n\n"
                "We've identified your concern. Please consult a qualified legal professional "
                "in your area for personalised advice."
            )

        # Append bot turn to history
        st.session_state.chat_history.append({
            "role": "bot",
            "text": bot_response,
            "category": predicted_category,
        })

# ─────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("#### 💬 Conversation")

    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {turn["text"]}</div>',
                unsafe_allow_html=True
            )
        else:
            category_display = turn.get("category", "").replace("_", " ")
            st.markdown(
                f'<div class="bot-bubble">'
                f'<span class="category-tag">⚖️ {category_display}</span><br>'
                f'{turn["text"]}'
                f'</div>',
                unsafe_allow_html=True
            )

    # Clear conversation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Pocket Legal Assistant · Academic Project · Not a substitute for professional legal advice
</div>
""", unsafe_allow_html=True)
