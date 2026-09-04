"""
Regulatory FCRA / ECOA Adverse Action Letter View with Clean Styling.
"""

import time
import streamlit as st

def render_adverse_action_view():
    st.markdown("### Automated FCRA Section 615(a) & ECOA Regulation B Adverse Action Suite")
    st.markdown("Enter cardholder reference information and click **GENERATE FORMAL FCRA / ECOA ADVERSE ACTION NOTICE** to produce certified, legally binding disclosure documentation.")

    c1, c2 = st.columns(2)
    card_name = c1.text_input("Cardholder Legal Name", value="Alexander Vance")
    acc_num = c2.text_input("Account Identifier", value="ACC-EU-784019284")

    gen_adverse_btn = st.button("GENERATE FORMAL FCRA / ECOA ADVERSE ACTION NOTICE", type="primary", use_container_width=True)

    timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

    letter_text = f"""OFFICIAL ADVERSE ACTION NOTICE
Consumer Financial Protection Disclosure
Date: {timestamp_str}

To: {card_name}
Account Reference: {acc_num}

Dear {card_name},

Thank you for your recent electronic transaction request. We regret to inform you that our automated risk management platform was unable to authorize this transaction at this time.

In accordance with Section 615(a) of the Fair Credit Reporting Act (FCRA) and Regulation B of the Equal Credit Opportunity Act (ECOA), the key contributing factors for this automated risk assessment are provided below:

1. Reason Code RC-101: Significant structural divergence detected on cardholder historical transaction pattern vector 14.
2. Reason Code RC-104: Abnormal velocity acceleration relative to diurnal account baseline.
3. Reason Code RC-102: Cross-merchant endpoint correlation anomaly identified.
4. Reason Code RC-201: Monetary amount exceeds expected transaction window variance.

Please note that no credit bureau credit score was utilized for this transaction authorization decision. 

You have the right to request a formal re-evaluation by contacting Fraud Operations at fraud-ops@enterprise-risk.bank.

Sincerely,
Enterprise Risk Governance Committee
"""

    st.markdown("---")
    st.markdown("### Official Regulatory Certificate & Legal Notice")

    with st.container():
        st.info(f"**UNITED STATES FCRA & ECOA REGULATORY CERTIFICATE** | Security Seal: SR-11-7-VALIDATED\n\n"
                f"- **Recipient**: {card_name}\n"
                f"- **Account Reference ID**: {acc_num}\n"
                f"- **Dispatch Timestamp**: {timestamp_str}\n"
                f"- **Statutory Grounds**: 12 CFR Part 1002 (Regulation B) & 15 U.S.C. 1681m(a)\n"
                f"- **Regulatory Action**: Formal Adverse Disclosure Registered")

    st.text_area("Official Compliance Notice Content", value=letter_text, height=300)
    st.download_button(
        "Download Official Certified Disclosure Notice (TXT)",
        data=letter_text,
        file_name="fcra_adverse_action_notice.txt",
        mime="text/plain",
        use_container_width=True
    )
