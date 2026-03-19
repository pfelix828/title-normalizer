"""Generate synthetic job title data for training.

Creates messy, realistic job titles mapped to standardized seniority levels
and job functions. Mirrors the real-world problem of normalizing CRM contact
titles for segmentation and targeting.

Noise categories mirror actual CRM data quality issues:
- Abbreviations, typos, misspellings, truncation (field length limits)
- Geo/team/division qualifiers appended or prepended
- Credential and degree suffixes (MBA, CPA, PMP, etc.)
- Separator variation (commas, pipes, slashes, dashes)
- Word reordering and dropped words
- Creative/startup titles, dual-role combos, bare ambiguous titles
- Free-text fragments, copy-paste artifacts, encoding junk
"""

import random
import csv
from pathlib import Path

RANDOM_SEED = 42

# --- Taxonomy ---

SENIORITY_LEVELS = {
    "individual_contributor": 0,
    "senior": 1,
    "lead": 2,
    "manager": 3,
    "senior_manager": 4,
    "director": 5,
    "senior_director": 6,
    "vp": 7,
    "svp": 8,
    "c_suite": 9,
}

FUNCTIONS = {
    "engineering": 0,
    "data": 1,
    "product": 2,
    "marketing": 3,
    "sales": 4,
    "finance": 5,
    "hr": 6,
    "operations": 7,
    "design": 8,
    "legal": 9,
}

# --- Title templates ---
# Each entry: (seniority_key, function_key, list of canonical title forms)

TITLE_TEMPLATES = [
    # ==================== Engineering ====================
    ("individual_contributor", "engineering", [
        "Software Engineer", "Software Developer", "Developer", "Programmer",
        "Backend Engineer", "Frontend Engineer", "Full Stack Engineer",
        "Full Stack Developer", "Web Developer", "Systems Engineer",
        "Platform Engineer", "Infrastructure Engineer", "DevOps Engineer",
        "Site Reliability Engineer", "SRE", "QA Engineer", "Test Engineer",
        "Mobile Engineer", "iOS Engineer", "Android Engineer",
        # Creative/startup variants
        "Software Development Engineer", "Application Developer",
        "Cloud Engineer", "Embedded Engineer", "Firmware Engineer",
        "Build Engineer", "Release Engineer", "Security Engineer",
        "Automation Engineer", "Integration Engineer",
    ]),
    ("senior", "engineering", [
        "Senior Software Engineer", "Senior Developer",
        "Senior Backend Engineer", "Senior Frontend Engineer",
        "Senior Full Stack Engineer", "Senior Platform Engineer",
        "Senior Infrastructure Engineer", "Senior DevOps Engineer",
        "Senior SRE", "Senior QA Engineer", "Senior Mobile Engineer",
        "Staff Engineer", "Staff Software Engineer",
        "Senior Security Engineer", "Senior Cloud Engineer",
        "Software Engineer III", "SDE III", "SDE 3",
        "MTS", "Member of Technical Staff",
    ]),
    ("lead", "engineering", [
        "Lead Engineer", "Lead Software Engineer", "Lead Developer",
        "Tech Lead", "Technical Lead", "Engineering Lead",
        "Lead Backend Engineer", "Lead Frontend Engineer",
        "Principal Engineer", "Principal Software Engineer",
        "Architect", "Software Architect", "Solutions Architect",
        "Distinguished Engineer",
    ]),
    ("manager", "engineering", [
        "Engineering Manager", "Software Engineering Manager",
        "Development Manager", "QA Manager", "DevOps Manager",
        "IT Manager", "Software Development Manager",
    ]),
    ("senior_manager", "engineering", [
        "Senior Engineering Manager", "Senior Development Manager",
        "Senior IT Manager",
    ]),
    ("director", "engineering", [
        "Director of Engineering", "Engineering Director",
        "Director of Software Engineering", "Director of Development",
        "Director of QA", "Director of Platform Engineering",
        "Director of IT", "Director of Technology",
    ]),
    ("senior_director", "engineering", [
        "Senior Director of Engineering",
        "Senior Director of Software Engineering",
        "Senior Director of Technology",
    ]),
    ("vp", "engineering", [
        "VP of Engineering", "VP Engineering",
        "Vice President of Engineering", "Vice President Engineering",
        "VP of Technology", "VP Technology",
    ]),
    ("svp", "engineering", [
        "SVP of Engineering", "Senior Vice President of Engineering",
        "SVP of Technology",
    ]),
    ("c_suite", "engineering", [
        "CTO", "Chief Technology Officer", "Chief Technical Officer",
        "Chief Engineering Officer", "Chief Information Officer", "CIO",
    ]),

    # ==================== Data / Analytics / ML ====================
    ("individual_contributor", "data", [
        "Data Analyst", "Data Scientist", "Analytics Analyst",
        "Business Analyst", "BI Analyst", "Business Intelligence Analyst",
        "ML Engineer", "Machine Learning Engineer", "Analytics Engineer",
        "Data Engineer", "Research Scientist", "Quantitative Analyst",
        "Reporting Analyst", "Database Administrator", "DBA",
        "Data Warehouse Analyst", "ETL Developer", "BI Developer",
        "Data Operations Analyst", "Insights Analyst",
    ]),
    ("senior", "data", [
        "Senior Data Analyst", "Senior Data Scientist",
        "Senior Analytics Analyst", "Senior Business Analyst",
        "Senior BI Analyst", "Senior ML Engineer",
        "Senior Machine Learning Engineer", "Senior Analytics Engineer",
        "Senior Data Engineer", "Senior Research Scientist",
        "Staff Data Scientist", "Staff Data Engineer",
        "Senior Quantitative Analyst", "Senior BI Developer",
    ]),
    ("lead", "data", [
        "Lead Data Scientist", "Lead Data Analyst", "Lead Data Engineer",
        "Lead ML Engineer", "Lead Analytics Engineer",
        "Principal Data Scientist", "Principal Data Engineer",
        "Principal Analyst",
    ]),
    ("manager", "data", [
        "Data Science Manager", "Analytics Manager",
        "Data Analytics Manager", "BI Manager",
        "Data Engineering Manager", "ML Engineering Manager",
        "Research Manager",
    ]),
    ("senior_manager", "data", [
        "Senior Analytics Manager", "Senior Data Science Manager",
        "Senior Data Engineering Manager",
    ]),
    ("director", "data", [
        "Director of Data Science", "Director of Analytics",
        "Director of Data", "Director of Data Engineering",
        "Director of Business Intelligence",
        "Director of Machine Learning", "Director of Research",
    ]),
    ("senior_director", "data", [
        "Senior Director of Data Science",
        "Senior Director of Analytics", "Senior Director of Data",
    ]),
    ("vp", "data", [
        "VP of Data", "VP of Data Science", "VP of Analytics",
        "VP Data", "Vice President of Data Science",
        "VP of Business Intelligence",
    ]),
    ("svp", "data", [
        "SVP of Data", "Senior Vice President of Data Science",
    ]),
    ("c_suite", "data", [
        "CDO", "Chief Data Officer", "Chief Analytics Officer",
        "Chief AI Officer", "CAIO",
    ]),

    # ==================== Product ====================
    ("individual_contributor", "product", [
        "Product Manager", "Product Analyst", "Product Owner",
        "Associate Product Manager", "Technical Product Manager",
        "Product Specialist", "Product Coordinator",
    ]),
    ("senior", "product", [
        "Senior Product Manager", "Senior Technical Product Manager",
        "Senior Product Owner", "Staff Product Manager",
        "Product Manager II", "Product Manager III",
    ]),
    ("lead", "product", [
        "Lead Product Manager", "Principal Product Manager",
        "Group Product Manager",
    ]),
    ("manager", "product", ["Product Management Manager"]),
    ("director", "product", [
        "Director of Product", "Director of Product Management",
        "Product Director",
    ]),
    ("senior_director", "product", [
        "Senior Director of Product",
        "Senior Director of Product Management",
    ]),
    ("vp", "product", [
        "VP of Product", "VP Product", "Vice President of Product",
        "Vice President of Product Management",
    ]),
    ("svp", "product", [
        "SVP of Product", "Senior Vice President of Product",
    ]),
    ("c_suite", "product", ["CPO", "Chief Product Officer"]),

    # ==================== Marketing ====================
    ("individual_contributor", "marketing", [
        "Marketing Specialist", "Marketing Coordinator",
        "Marketing Analyst", "Content Marketer", "Content Specialist",
        "Demand Generation Specialist", "Growth Marketer",
        "SEO Specialist", "Email Marketing Specialist",
        "Social Media Manager", "Brand Specialist",
        "Marketing Associate", "Copywriter", "Content Writer",
        "Digital Marketer", "Marketing Assistant",
        "Events Coordinator", "PR Specialist",
        "Communications Specialist", "Media Buyer",
    ]),
    ("senior", "marketing", [
        "Senior Marketing Specialist", "Senior Marketing Analyst",
        "Senior Content Marketer", "Senior Demand Generation Specialist",
        "Senior Growth Marketer", "Senior Copywriter",
        "Senior Digital Marketer", "Senior PR Specialist",
    ]),
    ("lead", "marketing", [
        "Lead Marketing Specialist", "Marketing Team Lead",
        "Lead Content Strategist",
    ]),
    ("manager", "marketing", [
        "Marketing Manager", "Content Marketing Manager",
        "Demand Generation Manager", "Growth Manager", "Brand Manager",
        "Digital Marketing Manager", "Product Marketing Manager",
        "Communications Manager", "PR Manager", "Events Manager",
    ]),
    ("senior_manager", "marketing", [
        "Senior Marketing Manager", "Senior Product Marketing Manager",
        "Senior Demand Generation Manager",
    ]),
    ("director", "marketing", [
        "Director of Marketing", "Marketing Director",
        "Director of Demand Generation", "Director of Content Marketing",
        "Director of Product Marketing", "Director of Growth",
        "Director of Communications", "Director of Brand",
    ]),
    ("senior_director", "marketing", [
        "Senior Director of Marketing",
        "Senior Director of Product Marketing",
    ]),
    ("vp", "marketing", [
        "VP of Marketing", "VP Marketing",
        "Vice President of Marketing", "VP of Growth",
    ]),
    ("svp", "marketing", [
        "SVP of Marketing", "Senior Vice President of Marketing",
    ]),
    ("c_suite", "marketing", ["CMO", "Chief Marketing Officer"]),

    # ==================== Sales ====================
    ("individual_contributor", "sales", [
        "Account Executive", "Sales Representative", "Sales Rep",
        "Business Development Representative", "BDR", "SDR",
        "Sales Development Representative", "Inside Sales Rep",
        "Account Manager", "Sales Associate", "Sales Consultant",
        "Territory Manager", "Sales Engineer", "Solutions Consultant",
        "Client Manager", "Relationship Manager",
        "Sales Coordinator", "Revenue Analyst",
    ]),
    ("senior", "sales", [
        "Senior Account Executive", "Senior Sales Representative",
        "Senior BDR", "Senior Account Manager",
        "Enterprise Account Executive", "Major Account Executive",
        "Strategic Account Executive", "Senior Sales Engineer",
        "Senior Solutions Consultant",
    ]),
    ("lead", "sales", [
        "Sales Team Lead", "Lead Account Executive",
        "Lead Sales Engineer",
    ]),
    ("manager", "sales", [
        "Sales Manager", "Account Management Manager",
        "Regional Sales Manager", "Inside Sales Manager",
        "Business Development Manager", "Territory Sales Manager",
        "Channel Manager", "Partner Manager",
    ]),
    ("senior_manager", "sales", [
        "Senior Sales Manager", "Senior Regional Sales Manager",
    ]),
    ("director", "sales", [
        "Director of Sales", "Sales Director",
        "Director of Business Development",
        "Director of Account Management",
        "Director of Revenue", "Director of Partnerships",
    ]),
    ("senior_director", "sales", ["Senior Director of Sales"]),
    ("vp", "sales", [
        "VP of Sales", "VP Sales", "Vice President of Sales",
        "VP of Revenue", "VP of Business Development",
    ]),
    ("svp", "sales", [
        "SVP of Sales", "Senior Vice President of Sales",
    ]),
    ("c_suite", "sales", ["CRO", "Chief Revenue Officer"]),

    # ==================== Finance ====================
    ("individual_contributor", "finance", [
        "Financial Analyst", "Accountant", "Staff Accountant",
        "Bookkeeper", "Accounts Payable Specialist",
        "Accounts Receivable Specialist", "FP&A Analyst",
        "Tax Analyst", "Treasury Analyst", "Audit Associate",
        "Payroll Specialist", "Billing Specialist",
        "Credit Analyst", "Collections Specialist",
        "Revenue Accountant", "Cost Analyst",
    ]),
    ("senior", "finance", [
        "Senior Financial Analyst", "Senior Accountant",
        "Senior FP&A Analyst", "Senior Tax Analyst",
        "Senior Auditor", "Senior Treasury Analyst",
    ]),
    ("lead", "finance", [
        "Lead Financial Analyst", "Lead Accountant",
    ]),
    ("manager", "finance", [
        "Finance Manager", "Accounting Manager", "FP&A Manager",
        "Tax Manager", "Controller", "Assistant Controller",
        "Audit Manager", "Treasury Manager", "Payroll Manager",
    ]),
    ("senior_manager", "finance", [
        "Senior Finance Manager", "Senior Accounting Manager",
    ]),
    ("director", "finance", [
        "Director of Finance", "Finance Director",
        "Director of FP&A", "Director of Accounting",
        "Director of Tax", "Director of Treasury",
    ]),
    ("senior_director", "finance", [
        "Senior Director of Finance",
    ]),
    ("vp", "finance", [
        "VP of Finance", "VP Finance",
        "Vice President of Finance", "Treasurer",
    ]),
    ("c_suite", "finance", ["CFO", "Chief Financial Officer"]),

    # ==================== HR / People ====================
    ("individual_contributor", "hr", [
        "HR Specialist", "HR Coordinator", "Recruiter",
        "Talent Acquisition Specialist", "People Operations Specialist",
        "HR Generalist", "Benefits Specialist", "Compensation Analyst",
        "HR Analyst", "HR Assistant", "Sourcer",
        "Onboarding Specialist", "Employee Relations Specialist",
        "HR Administrator", "Training Coordinator",
        "Learning & Development Specialist",
    ]),
    ("senior", "hr", [
        "Senior Recruiter", "Senior HR Specialist",
        "Senior Talent Acquisition Specialist", "Senior HR Generalist",
        "Senior People Operations Specialist",
        "Senior Compensation Analyst", "HR Business Partner",
    ]),
    ("lead", "hr", [
        "Lead Recruiter", "Recruiting Lead", "HR Team Lead",
    ]),
    ("manager", "hr", [
        "HR Manager", "Recruiting Manager",
        "Talent Acquisition Manager", "People Operations Manager",
        "Compensation Manager", "Benefits Manager",
        "Training Manager", "L&D Manager",
    ]),
    ("senior_manager", "hr", [
        "Senior HR Manager", "Senior Recruiting Manager",
    ]),
    ("director", "hr", [
        "Director of HR", "HR Director",
        "Director of Talent Acquisition",
        "Director of People Operations", "Director of Recruiting",
        "Director of People", "Director of L&D",
    ]),
    ("vp", "hr", [
        "VP of HR", "VP of People", "VP People",
        "Vice President of Human Resources",
        "VP of People Operations", "VP of Talent",
    ]),
    ("c_suite", "hr", [
        "CHRO", "Chief People Officer",
        "Chief Human Resources Officer",
    ]),

    # ==================== Operations ====================
    ("individual_contributor", "operations", [
        "Operations Analyst", "Operations Coordinator",
        "Operations Associate", "Project Coordinator",
        "Program Coordinator", "Supply Chain Analyst",
        "Logistics Coordinator", "Business Operations Analyst",
        "Process Analyst", "Procurement Specialist",
        "Buyer", "Administrative Assistant",
        "Office Manager", "Executive Assistant",
        "Facilities Coordinator",
    ]),
    ("senior", "operations", [
        "Senior Operations Analyst", "Senior Operations Associate",
        "Senior Project Coordinator", "Senior Buyer",
        "Senior Procurement Specialist",
    ]),
    ("lead", "operations", [
        "Lead Operations Analyst", "Lead Project Coordinator",
    ]),
    ("manager", "operations", [
        "Operations Manager", "Project Manager", "Program Manager",
        "Supply Chain Manager", "Logistics Manager",
        "Business Operations Manager", "Procurement Manager",
        "Facilities Manager", "Office Manager",
    ]),
    ("senior_manager", "operations", [
        "Senior Operations Manager", "Senior Program Manager",
        "Senior Project Manager",
    ]),
    ("director", "operations", [
        "Director of Operations", "Operations Director",
        "Director of Programs", "Director of Project Management",
        "Director of Supply Chain", "Director of Procurement",
    ]),
    ("vp", "operations", [
        "VP of Operations", "VP Operations",
        "Vice President of Operations",
        "VP of Supply Chain",
    ]),
    ("c_suite", "operations", ["COO", "Chief Operating Officer"]),

    # ==================== Design ====================
    ("individual_contributor", "design", [
        "Designer", "UX Designer", "UI Designer", "Product Designer",
        "Visual Designer", "Graphic Designer", "UX Researcher",
        "Interaction Designer", "Web Designer",
        "Motion Designer", "Brand Designer", "UI/UX Designer",
        "User Experience Designer", "User Interface Designer",
        "Design Researcher", "Content Designer",
    ]),
    ("senior", "design", [
        "Senior Designer", "Senior UX Designer", "Senior UI Designer",
        "Senior Product Designer", "Senior UX Researcher",
        "Staff Designer", "Staff Product Designer",
        "Senior Visual Designer", "Senior Graphic Designer",
    ]),
    ("lead", "design", [
        "Lead Designer", "Lead Product Designer",
        "Lead UX Designer", "Principal Designer",
        "Design Lead",
    ]),
    ("manager", "design", [
        "Design Manager", "UX Manager", "Design Team Manager",
        "Creative Manager",
    ]),
    ("director", "design", [
        "Director of Design", "Design Director", "Director of UX",
        "Director of Product Design", "Creative Director",
    ]),
    ("vp", "design", [
        "VP of Design", "VP Design", "Vice President of Design",
    ]),
    ("c_suite", "design", ["Chief Design Officer"]),

    # ==================== Legal ====================
    ("individual_contributor", "legal", [
        "Paralegal", "Legal Assistant", "Legal Analyst",
        "Compliance Analyst", "Contract Specialist",
        "Legal Coordinator", "Legal Secretary",
        "Regulatory Analyst", "IP Analyst",
        "Contracts Administrator",
    ]),
    ("senior", "legal", [
        "Senior Paralegal", "Senior Legal Analyst",
        "Senior Compliance Analyst", "Attorney", "Counsel",
        "Associate Counsel", "Corporate Counsel",
        "Senior Attorney", "Employment Counsel",
        "Privacy Counsel", "IP Counsel",
    ]),
    ("lead", "legal", [
        "Lead Counsel", "Senior Counsel",
    ]),
    ("manager", "legal", [
        "Legal Manager", "Compliance Manager", "Contracts Manager",
        "Regulatory Manager",
    ]),
    ("director", "legal", [
        "Director of Legal", "Legal Director",
        "Director of Compliance", "Director of Regulatory Affairs",
    ]),
    ("vp", "legal", [
        "VP of Legal", "VP Legal", "Vice President of Legal",
        "General Counsel", "Deputy General Counsel",
    ]),
    ("c_suite", "legal", ["CLO", "Chief Legal Officer"]),
]

# --- Variation rules ---

ABBREVIATIONS = {
    "Senior": ["Sr.", "Sr", "Snr", "Snr."],
    "Junior": ["Jr.", "Jr", "Jnr"],
    "Director": ["Dir.", "Dir"],
    "Manager": ["Mgr", "Mgr."],
    "Vice President": ["VP", "V.P."],
    "Engineer": ["Eng", "Eng.", "Engr"],
    "Specialist": ["Spec", "Spec."],
    "Coordinator": ["Coord", "Coord."],
    "Associate": ["Assoc", "Assoc."],
    "Representative": ["Rep", "Rep."],
    "Development": ["Dev", "Dev."],
    "Operations": ["Ops"],
    "Management": ["Mgmt"],
    "Technical": ["Tech"],
    "Business": ["Biz"],
    "Human Resources": ["HR"],
    "Information": ["Info"],
    "Administrator": ["Admin"],
    "Executive": ["Exec"],
    "Assistant": ["Asst", "Asst."],
    "Analyst": ["Anlst"],
    "Marketing": ["Mktg", "Mkt"],
    "Engineering": ["Eng", "Engg"],
    "Financial": ["Fin"],
    "Communications": ["Comms"],
    "Acquisition": ["Acq"],
}

FILLER_WORDS = ["of", "the", "&", "and", "-", ",", "/", "–"]
LEVEL_SUFFIXES = [
    "I", "II", "III", "IV", "V",
    "1", "2", "3", "4",
    "Level 1", "Level 2", "Level 3", "Level 4",
    "L3", "L4", "L5", "L6", "L7",
    "E3", "E4", "E5", "E6",
    "T3", "T4", "T5", "T6",
    "IC2", "IC3", "IC4", "IC5",
]
TEAM_PREFIXES = [
    "Team", "Group", "Global", "Regional", "Corporate", "Enterprise",
    "North America", "EMEA", "APAC", "LATAM",
    "US", "Americas", "International",
]

# CRM-realistic junk
GEO_QUALIFIERS = [
    "(EMEA)", "(APAC)", "(NA)", "(LATAM)", "(US)", "(Global)",
    "(West)", "(East)", "(Central)", "(Remote)",
    "- EMEA", "- APAC", "- North America", "- US", "- Global",
    "- West Region", "- East Coast", "- Remote",
    ", EMEA", ", North America", ", Global", ", US",
]

TEAM_QUALIFIERS = [
    "(Platform)", "(Infrastructure)", "(Growth)", "(Core)",
    "(Payments)", "(Ads)", "(Search)", "(Cloud)", "(Mobile)",
    "(Enterprise)", "(Consumer)", "(Internal Tools)", "(API)",
    "- Platform Team", "- Growth Team", "- Core Team",
    "- Payments", "- Commerce", "- Marketplace",
    ", Platform", ", Infrastructure", ", Payments",
]

CREDENTIAL_SUFFIXES = [
    ", MBA", ", CPA", ", PMP", ", PHR", ", SHRM-CP",
    ", CFA", ", JD", ", PhD", ", CISSP", ", PE",
    ", CISA", ", CISM", ", Esq.", ", CMA",
    " MBA", " CPA", " PMP", " CFA", " JD",
]

COMMON_MISSPELLINGS = {
    "Engineer": ["Engineeer", "Enginer", "Enginere", "Engneer"],
    "Manager": ["Manger", "Maneger", "Managr", "Mnager"],
    "Director": ["Direcotr", "Diretor", "Driector", "Direcor"],
    "Analyst": ["Analsyt", "Anlyst", "Analyts", "Analys"],
    "Specialist": ["Specilaist", "Specialst", "Specalist", "Spcialist"],
    "Coordinator": ["Coordiantor", "Coodinator", "Coordnator"],
    "Development": ["Developement", "Devlopment", "Develpoment"],
    "Marketing": ["Marekting", "Markting", "Marketng"],
    "Software": ["Sofware", "Sotfware", "Softwrae", "Softare"],
    "Operations": ["Opertions", "Operatoins", "Opeartions"],
    "President": ["Presidnet", "Persident", "Presdient"],
    "Senior": ["Senoir", "Senioe", "Snior", "Seior"],
    "Financial": ["Finanical", "Financal", "Finacial"],
    "Business": ["Busines", "Buisness", "Bussiness", "Busness"],
    "Technology": ["Technolgy", "Techology", "Technoloy"],
    "Acquisition": ["Aquisition", "Acqusition", "Acquistion"],
    "Intelligence": ["Intellegence", "Inteligence", "Intellignece"],
    "Compliance": ["Compiance", "Complance", "Complience"],
    "Account": ["Acount", "Acconut", "Accont"],
    "Product": ["Prodcut", "Prduct", "Porduct"],
}

# Separator variations people actually type in CRM fields
SEPARATOR_CHARS = [" / ", " | ", " - ", ", ", " & ", " -- ", " // ", " – "]


def apply_abbreviation(title: str, rng: random.Random) -> str:
    """Randomly abbreviate words in the title."""
    for word, abbrevs in ABBREVIATIONS.items():
        if word in title and rng.random() < 0.5:
            title = title.replace(word, rng.choice(abbrevs), 1)
    return title


def apply_case_variation(title: str, rng: random.Random) -> str:
    """Randomly change casing."""
    r = rng.random()
    if r < 0.10:
        return title.upper()
    elif r < 0.20:
        return title.lower()
    elif r < 0.25:
        return title.title()
    elif r < 0.30:
        # Random mixed case — real CRM copy-paste artifact
        return "".join(
            c.upper() if rng.random() < 0.3 else c.lower()
            for c in title
        )
    return title


def apply_filler(title: str, rng: random.Random) -> str:
    """Randomly insert or remove filler words."""
    if " of " in title and rng.random() < 0.4:
        title = title.replace(" of ", " ", 1)
    elif " of " not in title and rng.random() < 0.2:
        words = title.split()
        if len(words) >= 2:
            pos = rng.randint(1, len(words) - 1)
            filler = rng.choice(["of", "&", "and", "-", "/"])
            words.insert(pos, filler)
            title = " ".join(words)
    return title


def apply_level_suffix(title: str, seniority: str, rng: random.Random) -> str:
    """Add level suffixes — common in tech companies."""
    if seniority in ("individual_contributor", "senior", "lead") and rng.random() < 0.25:
        title = f"{title} {rng.choice(LEVEL_SUFFIXES)}"
    return title


def apply_prefix(title: str, rng: random.Random) -> str:
    """Add team/scope/geo prefixes."""
    if rng.random() < 0.12:
        title = f"{rng.choice(TEAM_PREFIXES)} {title}"
    return title


def apply_typo(title: str, rng: random.Random) -> str:
    """Introduce realistic typos — swaps, deletions, doubles, common misspellings."""
    # Try common misspelling first (more realistic)
    for word, misspells in COMMON_MISSPELLINGS.items():
        if word in title and rng.random() < 0.15:
            title = title.replace(word, rng.choice(misspells), 1)
            return title

    if len(title) <= 5:
        return title

    r = rng.random()
    chars = list(title)

    if r < 0.3:
        # Character swap
        pos = rng.randint(1, len(chars) - 2)
        if chars[pos].isalpha() and chars[pos + 1].isalpha():
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    elif r < 0.55:
        # Character deletion
        pos = rng.randint(1, len(chars) - 2)
        if chars[pos].isalpha():
            chars.pop(pos)
    elif r < 0.75:
        # Character doubling
        pos = rng.randint(1, len(chars) - 2)
        if chars[pos].isalpha():
            chars.insert(pos, chars[pos])
    else:
        # Adjacent key hit (simple approximation)
        adjacent = {
            "a": "sq", "s": "ad", "d": "sf", "e": "wr", "r": "et",
            "t": "ry", "i": "uo", "o": "ip", "n": "bm", "g": "fh",
        }
        pos = rng.randint(1, len(chars) - 2)
        c = chars[pos].lower()
        if c in adjacent:
            chars[pos] = rng.choice(list(adjacent[c]))

    return "".join(chars)


def apply_whitespace_noise(title: str, rng: random.Random) -> str:
    """Add extra spaces, tabs, or trim inconsistently."""
    r = rng.random()
    if r < 0.15:
        title = "  " + title
    if rng.random() < 0.15:
        title = title + "  "
    if rng.random() < 0.10:
        # Double space between random words
        words = title.split()
        if len(words) >= 2:
            pos = rng.randint(0, len(words) - 2)
            words[pos] = words[pos] + "  "
            title = " ".join(words)
    if rng.random() < 0.05:
        # Tab character (real CRM export artifact)
        title = title.replace(" ", "\t", 1)
    return title


def apply_geo_qualifier(title: str, rng: random.Random) -> str:
    """Append geographic region — very common in CRM exports."""
    if rng.random() < 0.5:
        title = f"{title} {rng.choice(GEO_QUALIFIERS)}"
    return title


def apply_team_qualifier(title: str, rng: random.Random) -> str:
    """Append team/division name — common in large companies."""
    if rng.random() < 0.5:
        title = f"{title} {rng.choice(TEAM_QUALIFIERS)}"
    return title


def apply_credential_suffix(title: str, rng: random.Random) -> str:
    """Append credentials/degrees — people put these in title fields."""
    title = f"{title}{rng.choice(CREDENTIAL_SUFFIXES)}"
    return title


def apply_truncation(title: str, rng: random.Random) -> str:
    """Truncate title — CRM field length limits cut off data."""
    if len(title) > 15:
        # Cut at a random point, sometimes mid-word
        cut_point = rng.randint(max(10, len(title) // 2), len(title) - 2)
        title = title[:cut_point]
        # Sometimes add trailing artifacts
        if rng.random() < 0.3:
            title = title + "..."
        elif rng.random() < 0.2:
            title = title + ".."
    return title


def apply_reordering(title: str, rng: random.Random) -> str:
    """Reorder title words — 'Director of Engineering' → 'Engineering Director'."""
    if " of " in title:
        parts = title.split(" of ", 1)
        if len(parts) == 2 and len(parts[1].split()) <= 3:
            sep = rng.choice([" ", ", ", " - "])
            title = f"{parts[1]}{sep}{parts[0]}"
    elif rng.random() < 0.3:
        words = title.split()
        if 2 <= len(words) <= 4:
            rng.shuffle(words)
            title = " ".join(words)
    return title


def apply_separator_variation(title: str, rng: random.Random) -> str:
    """Replace spaces with various separators people type in CRM."""
    if " " in title:
        words = title.split()
        if len(words) >= 2:
            pos = rng.randint(0, len(words) - 2)
            sep = rng.choice(SEPARATOR_CHARS)
            title = " ".join(words[:pos + 1]) + sep + " ".join(words[pos + 1:])
    return title


def apply_word_drop(title: str, rng: random.Random) -> str:
    """Drop a word — lazy data entry or copy-paste error."""
    words = title.split()
    if len(words) >= 3:
        drop_idx = rng.randint(0, len(words) - 1)
        words.pop(drop_idx)
        title = " ".join(words)
    return title


def apply_encoding_artifact(title: str, rng: random.Random) -> str:
    """Add encoding artifacts from CRM exports — smart quotes, mojibake, etc."""
    replacements = [
        (" - ", " \u2013 "),   # en dash
        (" - ", " \u2014 "),   # em dash
        ("'", "\u2019"),       # smart quote
        ("&", "&amp;"),        # HTML entity
        (" ", "\u00a0"),       # non-breaking space
    ]
    old, new = rng.choice(replacements)
    if old in title:
        title = title.replace(old, new, 1)
    return title


def generate_messy_title(
    canonical: str, seniority: str, rng: random.Random, noise_level: float = 1.0
) -> str:
    """Apply random variations to a canonical title.

    Args:
        canonical: The clean title string
        seniority: Seniority level key (for context-dependent variations)
        rng: Random number generator for reproducibility
        noise_level: 0.0 = clean, 1.0 = full noise. Controls variation probability.
    """
    title = canonical

    # Core transformations (applied frequently)
    if rng.random() < 0.70 * noise_level:
        title = apply_abbreviation(title, rng)
    if rng.random() < 0.50 * noise_level:
        title = apply_filler(title, rng)
    if rng.random() < 0.40 * noise_level:
        title = apply_level_suffix(title, seniority, rng)
    if rng.random() < 0.35 * noise_level:
        title = apply_case_variation(title, rng)

    # Structural transforms (moderate frequency)
    if rng.random() < 0.25 * noise_level:
        title = apply_reordering(title, rng)
    if rng.random() < 0.20 * noise_level:
        title = apply_prefix(title, rng)
    if rng.random() < 0.20 * noise_level:
        title = apply_separator_variation(title, rng)
    if rng.random() < 0.15 * noise_level:
        title = apply_word_drop(title, rng)

    # CRM junk (less frequent but impactful)
    if rng.random() < 0.15 * noise_level:
        title = apply_geo_qualifier(title, rng)
    if rng.random() < 0.12 * noise_level:
        title = apply_team_qualifier(title, rng)
    if rng.random() < 0.10 * noise_level:
        title = apply_credential_suffix(title, rng)
    if rng.random() < 0.08 * noise_level:
        title = apply_truncation(title, rng)

    # Surface noise (applied broadly)
    if rng.random() < 0.25 * noise_level:
        title = apply_typo(title, rng)
    if rng.random() < 0.20 * noise_level:
        title = apply_whitespace_noise(title, rng)
    if rng.random() < 0.08 * noise_level:
        title = apply_encoding_artifact(title, rng)

    return title


def generate_dataset(
    n_samples: int = 20000,
    noise_level: float = 1.0,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Generate a dataset of messy titles with labels.

    Args:
        n_samples: Number of title samples to generate
        noise_level: How messy the titles should be (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: raw_title, canonical_title, seniority, function,
        seniority_id, function_id
    """
    rng = random.Random(seed)
    records = []

    for _ in range(n_samples):
        template = rng.choice(TITLE_TEMPLATES)
        seniority_key, function_key, canonical_titles = template
        canonical = rng.choice(canonical_titles)
        raw = generate_messy_title(canonical, seniority_key, rng, noise_level)

        records.append({
            "raw_title": raw,
            "canonical_title": canonical,
            "seniority": seniority_key,
            "function": function_key,
            "seniority_id": SENIORITY_LEVELS[seniority_key],
            "function_id": FUNCTIONS[function_key],
        })

    return records


def save_dataset(records: list[dict], output_dir: str = "data") -> None:
    """Save generated records to CSV."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / "titles.csv"
    fieldnames = ["raw_title", "canonical_title", "seniority", "function", "seniority_id", "function_id"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} records to {filepath}")


if __name__ == "__main__":
    records = generate_dataset(n_samples=20000)
    save_dataset(records)

    # Print sample
    print("\nSample titles:")
    rng = random.Random(99)
    for r in rng.sample(records, 10):
        print(f"  {r['raw_title']:<55} → {r['seniority']:<20} | {r['function']}")
