"""Fictional HR Document Generator for Enterprise HR & Policy Navigator.

Generates realistic sample enterprise policy PDFs with controlled page counts,
version metadata, cross-document overrides, complex layouts, tables, and scanned pages.
"""

from pathlib import Path
from typing import List, Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    FrameBreak,
    BaseDocTemplate,
    PageTemplate,
    Frame,
)
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically for running footers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header
        self.drawString(54, letter[1] - 36, "Enterprise HR Global Policy Management System")
        self.drawRightString(letter[0] - 54, letter[1] - 36, "CONFIDENTIAL & PROPRIETARY")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer
        self.line(54, 46, letter[0] - 54, 46)
        self.drawString(54, 32, "Subject to corporate revision and jurisdictional regulations")
        self.drawRightString(
            letter[0] - 54,
            32,
            f"Page {self._pageNumber} of {page_count}",
        )
        self.restoreState()


def get_custom_styles() -> Dict[str, ParagraphStyle]:
    """Return dictionary of configured paragraph styles."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )
    alert_style = ParagraphStyle(
        "AlertBox",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#9B2C2C"),
        backColor=colors.HexColor("#FFF5F5"),
        borderColor=colors.HexColor("#FEB2B2"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=8,
        spaceAfter=8,
    )

    return {
        "title": title_style,
        "meta": meta_style,
        "h1": h1_style,
        "body": body_style,
        "alert": alert_style,
    }


def build_leave_policy_v1(output_path: Path) -> None:
    """Generate Leave_Policy_v1.pdf (20 days annual leave, effective 2024-01-01). 4 pages."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Page 1
    story.append(Paragraph("Enterprise Leave & Absence Policy", styles["title"]))
    story.append(
        Paragraph(
            "Document ID: POL-HR-LEAVE-001 | Version: 1.0 | Effective Date: 2024-01-01 | Owner: Corporate HR Operations",
            styles["meta"],
        )
    )
    story.append(Paragraph("Section 1: Policy Scope and Guiding Principles", styles["h1"]))
    story.append(
        Paragraph(
            "1.1 This policy governs all full-time and part-time permanent employees of the Enterprise. "
            "Our objective is to maintain a healthy work-life balance while ensuring operational continuity "
            "across all business units and international operational hubs.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "1.2 Compliance with statutory local labor legislation is mandatory. Where local statutory laws "
            "provide greater benefits than specified in this policy, local statutory mandates shall apply.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 2: Annual Paid Leave Entitlement", styles["h1"]))
    story.append(
        Paragraph(
            "2.1 All full-time employees are entitled to exactly twenty (20) paid annual leave days per calendar "
            "year, accruing on a monthly basis at the rate of 1.66 working days per completed calendar month of active service, "
            "effective January 1, 2024.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "2.2 New hires commence accrual on their official start date. Employees who start on or before the 15th of the month "
            "receive the full monthly accrual for that month.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Section 3: Sick, Medical, and Wellness Absences", styles["h1"]))
    story.append(
        Paragraph(
            "3.1 Full-time employees receive ten (10) days of paid sick leave annually, credited at the beginning "
            "of each calendar year on January 1. Sick leave does not roll over from year to year.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.2 For any single medical absence exceeding two (2) consecutive working days, an official medical "
            "certificate signed by a licensed medical practitioner must be uploaded to the HR portal within 48 hours of return.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.3 Habitual or patterned absences on Fridays or Mondays preceding or following public holidays may require "
            "medical certification regardless of duration, subject to People Operations review.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 4: Leave Application and Manager Approval Protocol", styles["h1"]))
    story.append(
        Paragraph(
            "4.1 All leave requests must be submitted through the internal HR portal at least two (2) weeks prior "
            "to planned vacations exceeding three consecutive days.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "4.2 Leave requests under three days require at least 48 hours advance notice to the direct reporting manager. "
            "Managers retain managerial discretion to deny leave requests during designated project freeze periods.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("Section 5: Carryover Limits, Rollover, and Forfeiture", styles["h1"]))
    story.append(
        Paragraph(
            "5.1 A maximum of five (5) unused annual leave days may be carried over from one calendar year into the next. "
            "Any accrued leave beyond five days remaining unused as of December 31 shall be permanently forfeited.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.2 All carried-over days must be utilized prior to March 31 of the new calendar year. Carried days not used "
            "by March 31 are automatically extinguished without financial compensation.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.3 Leave encashment is strictly prohibited during active employment. Upon voluntary separation, accrued and "
            "unused statutory leave is reimbursed in accordance with standard separation guidelines.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(Paragraph("Section 6: Parental, Bereavement, and Civic Duty Leave", styles["h1"]))
    story.append(
        Paragraph(
            "6.1 Primary parental caregivers are entitled to twelve (12) weeks of fully paid parental leave following "
            "the birth or adoption of a child. Secondary caregivers receive four (4) weeks of paid leave.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.2 Bereavement leave provides up to five (5) consecutive paid working days for immediate family members "
            "(spouse, child, parent, sibling) and up to two (2) days for extended family.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.3 Civic duty, including jury service and compulsory military call-ups, is supported with up to ten (10) days "
            "of paid time off upon presentation of official court or governmental summons.",
            styles["body"],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)


def build_leave_policy_v2(output_path: Path) -> None:
    """Generate Leave_Policy_v2.pdf (24 days annual leave, effective 2025-04-01, supersedes v1). 4 pages."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Page 1
    story.append(Paragraph("Enterprise Leave & Absence Policy (Revised)", styles["title"]))
    story.append(
        Paragraph(
            "Document ID: POL-HR-LEAVE-002 | Version: 2.0 | Effective Date: 2025-04-01 | Owner: Corporate HR Operations",
            styles["meta"],
        )
    )
    story.append(
        Paragraph(
            "IMPORTANT NOTICE OF SUPERSESSION: This document (Version 2.0) fully supersedes and revokes Leave Policy Version 1.0 "
            "in its entirety, effective April 1, 2025. All prior leave calculations and quotas established under Version 1.0 are null and void.",
            styles["alert"],
        )
    )
    story.append(Paragraph("Section 1: Modernized Workforce Well-being Framework", styles["h1"]))
    story.append(
        Paragraph(
            "1.1 Version 2.0 introduces modernized flexibility, mental recharge provisions, and an expanded annual paid "
            "leave quota reflecting competitive benchmark standards across enterprise technology companies.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 2: Revised Annual Paid Leave Entitlement (24 Days)", styles["h1"]))
    story.append(
        Paragraph(
            "2.1 Effective April 1, 2025, the standard paid annual leave entitlement for all full-time permanent employees "
            "is increased from 20 days to exactly twenty-four (24) paid annual leave days per calendar year.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "2.2 Monthly accrual rate is adjusted to 2.0 paid working days per completed calendar month. Employees with "
            "tenure exceeding five (5) continuous years receive a milestone bonus of two (2) supplementary loyalty leave days per year.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Section 3: Holistic Health and Mental Well-being Leave", styles["h1"]))
    story.append(
        Paragraph(
            "3.1 Employees are allocated twelve (12) fully paid health and wellness days per calendar year. This category covers "
            "acute illness, preventative medical care, mental health rejuvenation, and family dependent care.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.2 Medical certificates are now only requested when an individual sick leave episode extends beyond three (3) "
            "consecutive working days, streamlining administrative burden for brief illnesses.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.3 Annual 'Recharge Days': The enterprise designates two company-wide collective shutdown Fridays each year, "
            "which do not deduct from individual employee leave balances.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 4: Automated Self-Service Leave Protocol", styles["h1"]))
    story.append(
        Paragraph(
            "4.1 Standard leave requests up to five (5) working days are automatically auto-approved via the Workday/HRIS platform "
            "if submitted at least five business days in advance and no peer schedule conflicts exist.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "4.2 Vacations exceeding two weeks require formal Department Head approval at least thirty (30) days in advance "
            "to guarantee adequate operational redundancy.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("Section 5: Enhanced Rollover Limits and Extended Expiration", styles["h1"]))
    story.append(
        Paragraph(
            "5.1 Under Version 2.0, employees may carry forward up to eight (8) unused annual leave days into the subsequent calendar year "
            "(an increase from the previous five-day limitation).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.2 Carried-over leave must be taken before June 30 of the following year (extended from March 31). Any carried-over days "
            "remaining unspent after June 30 shall be forfeited without cash equivalent.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.3 Sabbatical Program: Employees with seven years of unblemished tenure may request a one-time unpaid sabbatical of up to "
            "three (3) months with guaranteed job reinstatement.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(Paragraph("Section 6: Expanded Family & Compassionate Leave", styles["h1"]))
    story.append(
        Paragraph(
            "6.1 Primary parental leave is expanded to sixteen (16) weeks at 100% base salary. Secondary caregivers receive six (6) weeks "
            "fully paid leave. This applies equally to natural births, surrogacy, and formal adoptions.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.2 Bereavement leave is standardized at seven (7) business days for immediate family members and three (3) days for extended family.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.3 Volunteer Service Time: All staff receive two (2) paid community service days annually to participate in verified non-profit activities.",
            styles["body"],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)


def build_wfh_policy(output_path: Path) -> None:
    """Generate WFH_Policy.pdf (Remote work guidelines and explicit overrides). 4 pages."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Page 1
    story.append(Paragraph("Flexible & Remote Work Policy (WFH)", styles["title"]))
    story.append(
        Paragraph(
            "Document ID: POL-HR-WFH-003 | Version: 1.2 | Effective Date: 2024-06-01 | Owner: People & Workplace Experience",
            styles["meta"],
        )
    )
    story.append(Paragraph("Section 1: Objective & Work Modalities", styles["h1"]))
    story.append(
        Paragraph(
            "1.1 This policy establishes rules and expectations for remote and hybrid work. We recognize that modern knowledge work "
            "can be performed effectively across distributed environments.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "1.2 Three official workplace classifications are established: (a) Office-First (4+ days on-site), "
            "(b) Hybrid Flexible (minimum 2 days on-site, up to 3 days remote), and (c) Designated Fully Remote.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 2: Hybrid Schedules & In-Office Presence", styles["h1"]))
    story.append(
        Paragraph(
            "2.1 Hybrid employees must coordinate their in-office days with their team leads. Tuesdays and Thursdays are designated "
            "as enterprise collaborative anchor days where on-site presence is strongly prioritized.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "2.2 Failure to maintain the required in-office cadence over a rolling six-week window may result in revocation of remote "
            "work privileges following HR consultation.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Section 3: Core Working Hours and Availability Standards", styles["h1"]))
    story.append(
        Paragraph(
            "3.1 All remote staff must remain accessible via Slack and corporate email during regional core hours: 10:00 AM to 4:00 PM local time.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.2 Video camera readiness is expected during team ceremonies, client presentations, and one-on-one performance discussions.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 4: Cross-Document Interaction and Leave Policy Override", styles["h1"]))
    story.append(
        Paragraph(
            "4.1 Standard Protocol Alignment: As explicitly referenced in Section 4 of the Leave Policy, any pre-planned time off during "
            "scheduled remote days must be officially filed via the HRIS portal.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "4.2 CRITICAL OVERRIDE RULE: Notwithstanding the annual leave entitlements defined in Section 2 of the Leave Policy, "
            "if an employee works remotely from an unapproved non-domestic jurisdiction or fails to obtain prior written authorization "
            "from their Vice President and Corporate Legal for out-of-state remote work exceeding fifteen (15) consecutive business days, "
            "all unauthorized remote working days shall be retroactively classified as unapproved absence and directly deducted from the employee's "
            "accumulated paid annual leave balance. If annual leave balance is zero, payroll deduction shall occur immediately.",
            styles["alert"],
        )
    )
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("Section 5: Home Office Equipment and Technology Stipend", styles["h1"]))
    story.append(
        Paragraph(
            "5.1 Full-time hybrid and remote personnel are entitled to a one-time setup reimbursement of up to $750 for ergonomic "
            "office furniture (chair, adjustable standing desk) upon submission of valid itemized receipts within 60 days of onboarding.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.2 IT Services issues a standardized hardware kit: laptop, dual monitors, enterprise dock, headset, and security key. "
            "All hardware remains corporate property and must be returned within 5 working days of employment termination.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.3 A monthly internet connectivity subsidy of $60 is disbursed automatically via regular payroll for approved remote staff.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(Paragraph("Section 6: Information Security, Data Protection & Ergonomics", styles["h1"]))
    story.append(
        Paragraph(
            "6.1 Employees are strictly prohibited from conducting confidential company operations on personal computers or unsecured public Wi-Fi "
            "without an active enterprise VPN connection and Multi-Factor Authentication (MFA).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.2 Family members or housemates must never be permitted to access or view corporate devices or customer records.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.3 Workplace safety guidelines extend to home offices. Employees must report any home-based occupational injuries to People Operations "
            "within 24 hours pursuant to workers' compensation statutory protocols.",
            styles["body"],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)


def build_code_of_conduct(output_path: Path) -> None:
    """Generate Code_of_Conduct.pdf (Ethics, anti-bribery, conflict of interest). 4 pages."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Page 1
    story.append(Paragraph("Enterprise Code of Business Conduct & Ethics", styles["title"]))
    story.append(
        Paragraph(
            "Document ID: POL-ETH-004 | Version: 3.0 | Effective Date: 2024-03-15 | Owner: Legal & Compliance Committee",
            styles["meta"],
        )
    )
    story.append(Paragraph("Section 1: Our Foundational Commitments and Values", styles["h1"]))
    story.append(
        Paragraph(
            "1.1 Our reputation is built upon integrity, transparency, and uncompromising ethical conduct. "
            "This Code applies to all directors, officers, employees, and contingent contractors across all global entities.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "1.2 Each employee is expected to exercise sound personal judgment, act with accountability, and speak up "
            "whenever ethical standards or regulatory compliance are compromised.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 2: Workplace Respect, Non-Discrimination & Harassment-Free Environment", styles["h1"]))
    story.append(
        Paragraph(
            "2.1 We maintain zero tolerance for harassment, bullying, discrimination, or offensive behavior based on race, "
            "gender, sexual orientation, disability, religion, age, veteran status, or genetic characteristics.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "2.2 Any employee subjected to or witnessing discriminatory conduct must promptly notify HR or submit a confidential "
            "report through the Integrity Helpline.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Section 3: Conflicts of Interest and Secondary Employment", styles["h1"]))
    story.append(
        Paragraph(
            "3.1 Employees must avoid any activity, investment, or association that interferes with their ability to exercise objective "
            "judgment in the performance of their duties for the enterprise.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.2 Dual employment, advisory roles, freelancing, or commercial consulting with any competitor, client, vendor, or industry partner "
            "is strictly forbidden without prior written authorization from the Chief Compliance Officer.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.3 Personal relationships involving direct or indirect reporting hierarchies must be disclosed immediately to People Operations "
            "to establish recusal procedures.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 4: Safeguarding Intellectual Property, Data & Confidential Information", styles["h1"]))
    story.append(
        Paragraph(
            "4.1 Proprietary source code, customer records, strategic financial plans, algorithmic designs, and marketing roadmaps "
            "constitute critical enterprise trade secrets.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "4.2 Confidential information must never be disclosed to third parties without an executed Non-Disclosure Agreement (NDA). "
            "This confidentiality obligation survives termination of employment indefinitely.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("Section 5: Anti-Bribery, Gifts, and Corporate Entertainment", styles["h1"]))
    story.append(
        Paragraph(
            "5.1 Strict compliance with the Foreign Corrupt Practices Act (FCPA), the UK Bribery Act, and applicable anti-corruption statutes "
            "is mandatory. Kickbacks, bribes, facilitation payments, or questionable gratuities are strictly prohibited.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.2 Nominal business gifts, promotional merchandise, or business meals valued under fifty US dollars ($50) may be accepted if "
            "customary, infrequent, and transparent. Any gift exceeding $50 in value must be formally reported to Compliance.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.3 Providing any gift, meal, or travel hospitality to government officials or regulators requires prior written pre-clearance "
            "from the General Counsel.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(Paragraph("Section 6: Whistleblower Protection and Reporting Channels", styles["h1"]))
    story.append(
        Paragraph(
            "6.1 The enterprise guarantees strict protection against retaliation for any employee who reports suspected wrongdoing in good faith. "
            "Retaliation is a severe violation resulting in immediate disciplinary termination.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.2 Reporting options include: (a) Direct Manager, (b) People Operations Partner, (c) Corporate Legal Counsel, "
            "or (d) The 24/7 Anonymous Ethics Hotline at +1-800-555-ETHIC or online at ethics.enterpriseportal.internal.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.3 The Audit and Compliance Committee investigates all reported complaints thoroughly, objectively, and confidentially.",
            styles["body"],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)


def build_benefits_insurance_policy(output_path: Path) -> None:
    """Generate Benefits_Insurance_Policy.pdf (Coverage tables, insurance tiers). 4 pages."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = get_custom_styles()
    story = []

    # Page 1
    story.append(Paragraph("Employee Group Benefits & Healthcare Policy", styles["title"]))
    story.append(
        Paragraph(
            "Document ID: POL-HR-BEN-005 | Version: 2.1 | Effective Date: 2024-05-01 | Owner: Total Rewards & Benefits",
            styles["meta"],
        )
    )
    story.append(Paragraph("Section 1: Eligibility and Enrollment Timelines", styles["h1"]))
    story.append(
        Paragraph(
            "1.1 All regular full-time employees scheduled to work at least 30 hours per week are eligible for medical, dental, "
            "vision, life, and disability benefits commencing on the first day of the calendar month following their hire date.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "1.2 New employees must complete electronic benefits elections within thirty (30) calendar days of their start date. "
            "Failure to complete timely enrollment results in automatic assignment to the Core HDHP default plan for employee-only coverage.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 2: Health Insurance Plans Comparison", styles["h1"]))
    story.append(
        Paragraph(
            "The enterprise sponsors three medical plan alternatives administered by BlueCross BlueShield / United Healthcare. "
            "Coverage amounts, deductibles, and out-of-pocket maximums are summarized in the table below:",
            styles["body"],
        )
    )

    # Table of coverage amounts
    table_data = [
        ["Plan Tier", "Annual Deductible\n(Indiv / Family)", "Out-of-Pocket Max\n(Indiv / Family)", "Primary Care\nCo-Pay", "Specialist\nCo-Pay", "Inpatient Hospital\nCoinsurance"],
        ["Premier PPO", "$500 / $1,000", "$3,000 / $6,000", "$20 co-pay", "$40 co-pay", "10% after deductible"],
        ["Standard PPO", "$1,250 / $2,500", "$4,500 / $9,000", "$30 co-pay", "$55 co-pay", "20% after deductible"],
        ["Core HDHP + HSA", "$2,000 / $4,000", "$6,000 / $12,000", "0% after ded.", "0% after ded.", "15% after deductible"],
    ]
    t = Table(table_data, colWidths=[1.1 * inch, 1.3 * inch, 1.3 * inch, 0.9 * inch, 0.9 * inch, 1.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF2F7")]),
            ]
        )
    )
    story.append(t)
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Section 3: Dental and Vision Insurance Schedules", styles["h1"]))
    story.append(
        Paragraph(
            "3.1 Dental coverage covers 100% of preventative dental exams, cleanings, and X-rays bi-annually. Basic restorative services "
            "(fillings, extractions) are reimbursed at 80%, and major prosthodontics at 50% up to an annual maximum benefit of $2,000 per member.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3.2 Vision insurance covers one comprehensive annual eye examination with a $10 co-pay and provides a $200 annual hardware "
            "allowance for frames or prescription contact lenses.",
            styles["body"],
        )
    )
    story.append(Paragraph("Section 4: Life and Accidental Death & Dismemberment (AD&D)", styles["h1"]))
    story.append(
        Paragraph(
            "4.1 Basic Group Term Life Insurance is provided at zero cost to employees at 2.0 times basic annual earnings, up to a maximum "
            "guaranteed benefit of $500,000. Equal coverage applies to Accidental Death and Dismemberment (AD&D).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "4.2 Voluntary supplemental life insurance can be purchased up to an additional 5x salary, capped at $1,000,000 subject to "
            "Evidence of Insurability (EOI) approval.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("Section 5: Short-Term and Long-Term Disability Protections", styles["h1"]))
    story.append(
        Paragraph(
            "5.1 Short-Term Disability (STD): Company-funded STD provides 66.67% of pre-disability weekly earnings up to $2,500/week, "
            "commencing after a seven-calendar-day elimination period for up to twenty-six (26) consecutive weeks.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "5.2 Long-Term Disability (LTD): LTD commences after 180 days of continuous disabling impairment, replacing 60% of basic monthly "
            "income up to a maximum of $10,000 per month until Social Security Normal Retirement Age.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(Paragraph("Section 6: Health Savings Accounts (HSA) & Wellness Stipends", styles["h1"]))
    story.append(
        Paragraph(
            "6.1 Employees selecting the Core HDHP receive an employer contribution of $750 (Individual) or $1,500 (Family) into their "
            "Health Savings Account (HSA), disbursed across semi-monthly payroll cycles.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.2 Annual Wellness Reimbursement: Employees are eligible for up to $500 annually for eligible wellness activities, "
            "including gym memberships, marathon registrations, fitness trackers, and registered mindfulness apps.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "6.3 Employee Assistance Program (EAP): Offers confidential, 24/7 counseling services (up to eight free sessions per issue) "
            "for employees and household family members.",
            styles["body"],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)


def build_messy_handbook(output_path: Path) -> None:
    """Generate Messy_Handbook.pdf (Two-column layout, repeated headers/footers, hyphenation, messy table). 4 pages."""
    # Define a two-column BaseDocTemplate
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54,
    )

    # Frame 1 (left column), Frame 2 (right column)
    margin = 36
    gutter = 18
    col_width = (letter[0] - 2 * margin - gutter) / 2
    frame_height = letter[1] - 108

    frame_left = Frame(margin, 54, col_width, frame_height, id="col1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_right = Frame(margin + col_width + gutter, 54, col_width, frame_height, id="col2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    template = PageTemplate(id="two_column", frames=[frame_left, frame_right])
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()

    messy_h1 = ParagraphStyle(
        "MessyH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#742A2A"),
        spaceBefore=8,
        spaceAfter=4,
    )
    messy_body = ParagraphStyle(
        "MessyBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.black,
        spaceAfter=6,
    )

    story = []

    # Column 1 / Page 1
    story.append(Paragraph("EMPLOYEE OPERATIONS HANDBOOK - GENERAL GUIDELINES (UNOFFICIAL REVISION)", messy_h1))
    story.append(
        Paragraph(
            "This hand-book sum-marizes gen-eral work-place stan-dards for all depart-ments. "
            "All em-ployees must com-ply with safe-ty reg-ulations, or-ganizational code, and su-pervisory di-rectives.",
            messy_body,
        )
    )
    story.append(Paragraph("SECTION 1: WORKPLACE ATTIRE & UNIFORM DRESS CODE", messy_h1))
    story.append(
        Paragraph(
            "Em-ployees rep-resent the com-pany's pub-lic im-age. Busi-ness cas-ual attire is re-quired Mon-day through Thurs-day. "
            "Jeans with-out fray-ing or dis-tressing are per-mitted on Fri-days only. "
            "Open-toed foot-wear and ath-letic gar-ments are strictly pro-hibited on pro-duction and lab-oratory floor fa-cilities.",
            messy_body,
        )
    )
    story.append(Paragraph("SECTION 2: PARKING PERMITS & COMMUTER TRANSIT", messy_h1))
    story.append(
        Paragraph(
            "Ve-hicle park-ing on cam-pus re-quires an elec-tronic park-ing tag affixed to the lower driv-er-side wind-shield. "
            "Park-ing in vis-itor or hand-icapped bays with-out au-thorized reg-istration re-sults in im-mediate tow-ing at the "
            "em-ployee's sole ex-pense. EV charg-ing sta-tions have a four-hour max-imum con-nected limit.",
            messy_body,
        )
    )
    story.append(FrameBreak())

    # Column 2 / Page 1 (moves to next frame on same page)
    story.append(Paragraph("SECTION 3: EQUIPMENT MAINTENANCE & OFFICE UTILITIES", messy_h1))
    story.append(
        Paragraph(
            "Desks and com-puter work-stations must be cleared at the end of each busi-ness day in ac-cordance with our "
            "clean-desk pro-tocol. Heat-ing ap-pliances, space heat-ers, and cook-ing de-vices are banned from per-sonal cu-bicles.",
            messy_body,
        )
    )
    story.append(Paragraph("SECTION 4: PERFORMANCE EVALUATION CYCLE", messy_h1))
    story.append(
        Paragraph(
            "An-nual per-formance re-views are con-ducted in No-vember. Mid-year check-ins occur in June. "
            "Re-view rat-ings de-termine mer-it com-pen-sa-tion ad-justments and pro-mo-tion eli-gibility.",
            messy_body,
        )
    )
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("SECTION 5: DISCIPLINARY ACTIONS AND ESCALATIONS", messy_h1))
    story.append(
        Paragraph(
            "Vio-lations of work-place con-duct fol-low a pro-gressive dis-ciplinary frame-work: "
            "(1) In-formal ver-bal coun-seling, (2) For-mal writ-ten rep-rimand, "
            "(3) Per-formance Im-provement Plan (PIP) of 30 to 60 days, and (4) Sep-aration of em-ploy-ment.",
            messy_body,
        )
    )
    story.append(Paragraph("SECTION 6: BREAK ROOM AND PANTRY PROTOCOLS", messy_h1))
    story.append(
        Paragraph(
            "The com-mon re-frigerator is thor-oughly purged every Fri-day at 5:00 PM. "
            "Any un-labeled con-tainers will be dis-posed of with-out ex-ception to main-tain sani-tary con-ditions.",
            messy_body,
        )
    )
    story.append(FrameBreak())

    # Column 2 / Page 2
    story.append(Paragraph("SECTION 7: TRAVEL REIMBURSEMENTS SCHEDULE", messy_h1))
    story.append(
        Paragraph(
            "Busi-ness trav-el ex-penses must be doc-umented with or-iginal re-ceipts. "
            "Per diem meal lim-its are fixed at six-ty-five dol-lars per day for do-mestic travel.",
            messy_body,
        )
    )

    # Messy Table
    table_content = [
        ["Expense Tier", "Max Limit", "Approval Level"],
        ["Local Taxi / Rideshare", "$45 / day", "Self-certified"],
        ["Domestic Flight", "Economy Class", "Manager"],
        ["Lodging / Hotel", "$220 / night", "Director"],
        ["Client Dinner / Ent.", "$100 / person", "VP Pre-approval"],
    ]
    tb = Table(table_content, colWidths=[1.1 * inch, 1.0 * inch, 1.1 * inch])
    tb.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story.append(tb)
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("SECTION 8: BADGE ACCESS AND SECURITY PROTOCOLS", messy_h1))
    story.append(
        Paragraph(
            "Se-curity iden-tification badges must be worn vis-ibly above the waist at all times while on cor-porate prem-ises. "
            "Tail-gating or hold-ing open se-cure ac-cess doors for un-badged in-dividuals is a se-vere se-curity vi-olation.",
            messy_body,
        )
    )
    story.append(Paragraph("SECTION 9: LOST PROPERTY DISPOSITION", messy_h1))
    story.append(
        Paragraph(
            "Any found items must be turned in to Fa-cilities Man-agement on the first floor. "
            "Un-claimed items will be do-nated to char-ity after for-ty-five (45) days of stor-age.",
            messy_body,
        )
    )
    story.append(FrameBreak())

    # Column 2 / Page 3
    story.append(Paragraph("SECTION 10: WORKPLACE INCLEMENT WEATHER POLICY", messy_h1))
    story.append(
        Paragraph(
            "In case of se-vere weath-er emer-gencies, fa-cility clo-sures will be broad-cast via the Emer-gency Noti-fication Sys-tem "
            "by 6:00 AM. Em-ployees are ex-pected to work re-motely when phy-sical fa-cilities are closed.",
            messy_body,
        )
    )
    story.append(Paragraph("SECTION 11: SMOKING AND TOBACCO BAN", messy_h1))
    story.append(
        Paragraph(
            "All com-pany fa-cilities and sur-rounding cam-pus grounds are 100% to-bacco and va-por-free zones.",
            messy_body,
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(Paragraph("SECTION 12: SOLICITATIONS & BULLETIN BOARDS", messy_h1))
    story.append(
        Paragraph(
            "So-licitation of prod-ucts, serv-ices, or po-litical cam-paigns is pro-hibited on com-pany prop-erty or through "
            "cor-porate com-munication chan-nels.",
            messy_body,
        )
    )
    story.append(FrameBreak())

    # Column 2 / Page 4
    story.append(Paragraph("SECTION 13: ACKNOWLEDGMENT FORM", messy_h1))
    story.append(
        Paragraph(
            "All em-ployees must dig-itally sign the an-nual hand-book com-pliance af-firmation within thir-ty days of re-ceipt.",
            messy_body,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)


def build_scanned_notice(output_path: Path) -> None:
    """Generate Scanned_Notice.pdf (Pages rendered purely as bitmap images with NO text layer). 3 pages."""
    pages: List[Image.Image] = []
    width, height = 1200, 1600

    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_header = ImageFont.truetype("arial.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 19)
    except Exception:
        font_title = ImageFont.load_default(size=28)
        font_header = ImageFont.load_default(size=22)
        font_body = ImageFont.load_default(size=19)

    notices = [
        [
            "OFFICIAL CORPORATE HR MEMORANDUM",
            "REF: HR-NOT-2024-88 | CLASSIFICATION: CONFIDENTIAL",
            "DATE: AUGUST 14, 2024",
            "TO: ALL REGIONAL OFFICE EMPLOYEES",
            "FROM: EXECUTIVE VP OF HUMAN RESOURCES",
            "SUBJECT: EMERGENCY FACILITY CLOSURE & HAZARD CONTINUITY PROTOCOL",
            "",
            "1. PURPOSE AND IMMEDIATE ACTION",
            "This memorandum serves as mandatory operational guidance regarding severe emergency",
            "disruptions, public health crises, and environmental evacuation orders.",
            "",
            "2. REMOTE ACTIVATION MANDATE",
            "Upon declaration of a Tier 1 Facility Closure by the Executive Crisis Committee,",
            "all non-essential site staff must transition to fully remote status within 3 hours.",
            "Site badge access will be suspended electronically until health clearance is granted.",
            "",
            "3. CONTINUITY COMPENSATION",
            "Employees unable to perform roles remotely due to equipment or laboratory dependency",
            "shall receive standard base compensation for up to 10 consecutive business days",
            "without mandatory deduction from their accrued personal leave reserves.",
            "",
            "[OFFICIAL STAMP: OFFICE OF CORPORATE COMPLIANCE - SIGNED & ARCHIVED]",
        ],
        [
            "OFFICIAL CORPORATE HR MEMORANDUM (PAGE 2)",
            "REF: HR-NOT-2024-88 | EMERGENCY FACILITY CLOSURE",
            "",
            "4. ESSENTIAL ON-SITE PERSONNEL PROTOCOLS",
            "Designated Tier 1 Infrastructure Specialists (Server Operations, Physical Security,",
            "and Facilities Maintenance) must adhere to the rotational shift safety roster.",
            "Hazard differential pay equal to 1.5x regular hourly compensation will be credited",
            "for each emergency shift performed on-site during designated closure periods.",
            "",
            "5. COMMUNICATION CADENCE & DAILY CHECK-INS",
            "All employees are required to reply to daily SMS safety status pings by 09:30 AM.",
            "Managers must submit affirmative roll-call confirmations to HR Incident Command",
            "by no later than 11:00 AM daily.",
            "",
            "6. TRAVEL RESTRICTIONS",
            "All non-essential business travel to or from affected regional sectors is suspended.",
            "Expenses incurred due to unexpected travel delays will be honored through emergency accounts.",
            "",
            "[HR OPERATIONS AUDIT TRAIL: VERIFIED FOR IMMEDIATE CIRCULATION]",
        ],
        [
            "OFFICIAL CORPORATE HR MEMORANDUM (PAGE 3)",
            "REF: HR-NOT-2024-88 | CLOSURE PROTOCOLS & REOPENING",
            "",
            "7. PHYSICAL REOPENING & RETURN-TO-WORK CLEARANCE",
            "No employee may return to corporate facilities until official written clearance is",
            "broadcast via enterprise emergency broadcast channels.",
            "Deep sanitization and air filtration audits will be conducted before re-admittance.",
            "",
            "8. QUESTIONS & REPORTING HOTLINE",
            "Inquiries regarding this notice or personal hardship accommodations should be directed to:",
            "Emergency Operations Center: eoc-response@enterpriseportal.internal",
            "Toll-Free Emergency Dispatch: +1-800-555-0199 (Extension 4)",
            "",
            "Approved by:",
            "/s/ Eleanor Vance, Chief People Officer",
            "/s/ Marcus Thorne, Director of Corporate Security",
            "",
            "[PHYSICALLY SCANNED COPY - ORIGINAL SIGNATURES ON FILE IN ARCHIVES ROOM 4B]",
        ],
    ]

    for page_idx, lines in enumerate(notices):
        # Slightly off-white scanner background
        img = Image.new("RGB", (width, height), color=(252, 252, 250))
        draw = ImageDraw.Draw(img)

        # Draw simulated header border
        draw.line([(80, 70), (width - 80, 70)], fill=(120, 120, 120), width=2)
        draw.line([(80, height - 80), (width - 80, height - 80)], fill=(180, 180, 180), width=2)

        # Draw simulated "SCANNED" stamp watermark
        draw.rectangle([(width - 380, 90), (width - 80, 150)], outline=(180, 80, 80), width=2)
        draw.text((width - 360, 105), "SCANNED ARCHIVE", font=font_header, fill=(180, 80, 80))

        y = 100
        for line in lines:
            if not line:
                y += 24
                continue
            is_title = line.startswith("OFFICIAL CORPORATE")
            is_header = line.startswith("REF:") or line.startswith("SUBJECT:") or line.startswith("[") or line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5.") or line.startswith("6.") or line.startswith("7.") or line.startswith("8.")
            
            if is_title:
                draw.text((80, y), line, font=font_title, fill=(20, 20, 40))
                y += 38
            elif is_header:
                draw.text((80, y), line, font=font_header, fill=(30, 30, 70))
                y += 32
            else:
                draw.text((90, y), line, font=font_body, fill=(35, 35, 45))
                y += 28

        # Draw footer notes
        draw.text((80, height - 60), f"Document Ref: HR-NOT-2024-88 | Page {page_idx + 1} of 3 | Scanned via Xerox WorkCentre", font=font_body, fill=(120, 120, 120))

        pages.append(img)

    # Save as PDF where all pages are pure raster images (no text layer)
    pages[0].save(
        str(output_path),
        save_all=True,
        append_images=pages[1:],
        format="PDF",
        resolution=150.0,
    )


def generate_all_docs(target_dir: Path) -> None:
    """Generate all fictional HR documents into the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating fictional HR documents in: {target_dir.resolve()}")

    docs = [
        ("Leave_Policy_v1.pdf", build_leave_policy_v1),
        ("Leave_Policy_v2.pdf", build_leave_policy_v2),
        ("WFH_Policy.pdf", build_wfh_policy),
        ("Code_of_Conduct.pdf", build_code_of_conduct),
        ("Benefits_Insurance_Policy.pdf", build_benefits_insurance_policy),
        ("Messy_Handbook.pdf", build_messy_handbook),
        ("Scanned_Notice.pdf", build_scanned_notice),
    ]

    for filename, builder_func in docs:
        file_path = target_dir / filename
        builder_func(file_path)
        print(f"  [+] Created: {filename}")

    print("Successfully generated all sample documents.")


if __name__ == "__main__":
    from backend.config import get_settings

    cfg = get_settings()
    docs_directory = cfg.get_resolved_documents_dir()
    generate_all_docs(docs_directory)
