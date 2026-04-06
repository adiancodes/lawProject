import streamlit as st
import joblib, os

# ── 1. Page config ──────────────────────────────────────────
st.set_page_config(page_title="Pocket Legal Assistant", layout="centered")

# ── 2. CSS — minimal legal-document theme ───────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600&family=Inter:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#1a1a1a}
.stApp{background:#f7f5f0}
.block-container{max-width:760px!important;padding-top:3rem!important}
.stTextArea textarea{background:#fff!important;border:1px solid #b0a99e!important;border-radius:0!important;color:#1a1a1a!important;font-size:.93rem!important;padding:.85rem 1rem!important;box-shadow:none!important}
.stTextArea textarea:focus{border-color:#1a1a1a!important;box-shadow:none!important}
.stTextArea label{display:none!important}
.stButton>button{background:#0f1923!important;color:#fff!important;border:none!important;border-radius:0!important;font-size:.78rem!important;font-weight:600!important;letter-spacing:.12em!important;text-transform:uppercase!important;padding:.7rem 2.2rem!important}
.stButton>button:hover{background:#1e3a5f!important}
.stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown strong,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown a{color:#1a1a1a!important}
#MainMenu,footer,header{visibility:hidden}
</style>""", unsafe_allow_html=True)

# ── 3. Load trained model and vectorizer ────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("legal_model.pkl"):
        return None, None
    return joblib.load("legal_model.pkl"), joblib.load("vectorizer.pkl")

model, vectorizer = load_model()

# ── 4. Legal response dictionary ────────────────────────────
RESPONSES = {
    "Traffic_Harassment": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- Motor Vehicles Act, 1988
- IPC Section 294 — obscene acts in a public place
- IPC Section 506 — criminal intimidation
- Prevention of Corruption Act, 1988 — if a bribe was demanded

**Suggested Next Steps**
1. Note the officer's name, badge number, time, and location.
2. File a written complaint with the Senior Police Inspector or DCP (Traffic).
3. Report bribe demands to the State Anti-Corruption Bureau.
4. Call the Police Control Room if you feel unsafe.""",

    "Tenant_Rights": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- The Rent Control Act (state-specific variant)
- Transfer of Property Act, 1882
- IPC Section 441 — criminal trespass (unauthorised entry by landlord)

**Suggested Next Steps**
1. Do not vacate without a formal court eviction order.
2. Send a legal notice via registered post documenting all grievances.
3. File a complaint before the Rent Control Tribunal in your jurisdiction.
4. Preserve rent receipts, bank records, photos, and all written communication.""",

    "Cybercrime_Financial_Fraud": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- IT Act, 2000 — Section 66C (identity theft), Section 66D (impersonation)
- IPC Section 420 — cheating and inducing delivery of property
- IPC Section 468 — forgery for the purpose of cheating

**Suggested Next Steps**
1. Call your bank's fraud helpline immediately to block the account or card.
2. Report at https://cybercrime.gov.in or call the Cyber Crime Helpline: 1930.
3. File an FIR at the nearest Cyber Crime Cell with all transaction records.""",

    "Cybercrime_Harassment": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- IT Act, 2000 — Section 66E (privacy), Section 67 (obscene content online)
- IPC Section 354D — stalking, including cyber-stalking
- IPC Section 384 — extortion / blackmail

**Suggested Next Steps**
1. Do not comply with blackmail demands.
2. Screenshot all evidence before blocking the offender.
3. Report the account to the platform, then file at https://cybercrime.gov.in.""",

    "Consumer_Protection": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- Consumer Protection Act, 2019
- Food Safety and Standards Act, 2006 — for food-related complaints
- Legal Metrology Act, 2009 — for overcharging or hidden fees

**Suggested Next Steps**
1. Send a formal legal notice to the seller demanding resolution within 15 days.
2. Call the National Consumer Helpline: 1800-11-4000 or visit https://consumerhelpline.gov.in.
3. File before the District Consumer Disputes Redressal Commission (DCDRC).""",

    "Employment_Dispute": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- Industrial Disputes Act, 1947 — wrongful termination, retrenchment
- Payment of Wages Act, 1936 — unpaid or delayed salary
- POSH Act, 2013 — workplace sexual harassment
- EPF & Miscellaneous Provisions Act, 1952 — PF non-deposit

**Suggested Next Steps**
1. Secure all documentation: pay slips, contract, emails.
2. Send a formal legal notice to HR through a registered advocate.
3. File a complaint with the Labour Commissioner of your state.
4. For PF issues, raise a grievance at https://epfigms.gov.in or call 1800-118-005.""",

    "Property_Dispute": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- Transfer of Property Act, 1882
- Hindu Succession Act, 1956 / Indian Succession Act, 1925
- IPC Section 447 — criminal trespass
- IPC Sections 467 and 468 — forgery of property documents

**Suggested Next Steps**
1. Secure all original documents: sale deed, title deed, mutation records.
2. File for a stay order in civil court to prevent further encroachment or transfer.
3. File an FIR if trespass or forgery is involved.
4. For builder disputes, file with RERA of your state.""",

    "Public_Nuisance": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- IPC Section 268 — definition of public nuisance
- IPC Section 290 — punishment for public nuisance
- CrPC Section 133 — Magistrate's order to remove a public nuisance
- Noise Pollution (Regulation and Control) Rules, 2000

**Suggested Next Steps**
1. File a written complaint with the local police station or Executive Magistrate.
2. Report noise violations to the State Pollution Control Board.
3. Complain to the Municipal Corporation for infrastructure issues.
4. Any citizen can file a Section 133 CrPC petition before the District Magistrate.""",

    "Theft": """**Disclaimer:** General awareness only. Not professional legal advice.

---
**Relevant Law**
- IPC Section 378 — definition of theft
- IPC Section 379 — punishment for theft (up to 3 years, fine, or both)
- IPC Section 356 — assault in committing theft (snatching)
- IPC Section 380 — theft in a dwelling house

**Suggested Next Steps**
1. File an FIR at the nearest police station immediately.
2. For vehicle theft, notify the RTO and your insurer within 24 hours.
3. For mobile theft, report the IMEI number at https://ceir.gov.in to block the device.""",
}

# ── 5. Page header ──────────────────────────────────────────
st.markdown("""
<div style="border-bottom:2px solid #1a1a1a;padding-bottom:1.2rem;margin-bottom:1.4rem">
  <div style="font-size:.68rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#666;margin-bottom:.4rem">Academic Project &mdash; AI-Assisted Legal Guidance</div>
  <div style="font-family:'EB Garamond',Georgia,serif;font-size:2.6rem;font-weight:600;color:#0d0d0d;line-height:1.15">Pocket Legal Assistant</div>
  <div style="font-size:.85rem;color:#555;margin-top:.5rem">Describe your legal situation in plain language. The system will classify your case and return the relevant statute and suggested course of action.</div>
</div>
""", unsafe_allow_html=True)

# ── 6. Input form ───────────────────────────────────────────
st.markdown('<div style="font-size:.68rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#888;margin-bottom:.5rem">Describe your situation</div>', unsafe_allow_html=True)
user_input = st.text_area("scenario", placeholder="e.g. My landlord is refusing to return my security deposit and entered my flat without permission...", height=150, max_chars=2000)
submitted  = st.button("Analyse")

# ── 7. Predict and display result ───────────────────────────
if submitted:
    if not user_input.strip():
        st.warning("Please describe your situation before submitting.")
    elif model is None:
        st.error("Model not found. Run python train_model.py first.")
    else:
        category     = model.predict(vectorizer.transform([user_input]))[0]
        response     = RESPONSES.get(category, "Please consult a qualified legal professional.")
        display_name = category.replace("_", " ").title()

        st.markdown(f"""
        <div style="border-left:3px solid #0f1923;background:#fff;padding:1.6rem 2rem;margin-top:1.8rem">
          <div style="font-size:.65rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#888;margin-bottom:.3rem">Legal Classification</div>
          <div style="font-family:'EB Garamond',Georgia,serif;font-size:1.45rem;font-weight:600;color:#0f1923;border-bottom:1px solid #e0dbd3;padding-bottom:.8rem;margin-bottom:1rem">{display_name}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div style="color:#1a1a1a;font-size:.9rem;line-height:1.78;padding:.4rem 0 0 0">{response}</div>', unsafe_allow_html=True)

# ── 8. Footer ───────────────────────────────────────────────
st.markdown('<div style="margin-top:3rem;border-top:1px solid #c8c3ba;padding-top:1rem;font-size:.72rem;color:#999">Pocket Legal Assistant &mdash; Academic Project &mdash; This tool does not provide professional legal advice.</div>', unsafe_allow_html=True)
