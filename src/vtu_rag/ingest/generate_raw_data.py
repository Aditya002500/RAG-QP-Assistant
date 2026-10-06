"""Generates authentic VTU Syllabus and Previous-Year Question Paper PDFs for 5th Sem CSE.
Includes standard VTU CBCS examination formatting and a scanned page to exercise OCR fallback.
"""

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


def create_syllabus_pdf(
    file_path: Path,
    subject_code: str,
    subject_name: str,
    credits: int,
    modules: dict,
    textbooks: list[str],
):
    doc = pymupdf.open()
    # Page 1: Header and Module 1 & 2
    page1 = doc.new_page(width=595, height=842)  # A4
    header_text = f"""VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI
B.E. Computer Science and Engineering - 2022 Scheme
FIFTH SEMESTER B.E. SYLLABUS

Course Title: {subject_name}
Course Code: {subject_code}
Credits: {credits}
CIE Marks: 50 | SEE Marks: 50 | Exam Hours: 3

COURSE OBJECTIVES:
1. Provide comprehensive knowledge of {subject_name} principles and applications.
2. Formulate theoretical foundations and practical design strategies for VTU engineering students.
3. Prepare students for professional industry standards and technical problem solving.

MODULE 1:
{modules['module_1']}

MODULE 2:
{modules['module_2']}
"""
    page1.insert_text((40, 50), header_text, fontsize=10, fontname="helv")

    # Page 2: Module 3, 4, 5 and Textbooks
    page2 = doc.new_page(width=595, height=842)
    p2_text = f"""{subject_code} - {subject_name} (Continued)

MODULE 3:
{modules['module_3']}

MODULE 4:
{modules['module_4']}

MODULE 5:
{modules['module_5']}

TEXTBOOKS:
""" + "\n".join(f"{i+1}. {tb}" for i, tb in enumerate(textbooks)) + """

COURSE OUTCOMES (COs):
CO1: Understand core principles and mathematical models.
CO2: Design and analyze architectural workflows.
CO3: Implement algorithms and engineering solutions.
CO4: Evaluate performance metrics and quality benchmarks.
"""
    page2.insert_text((40, 50), p2_text, fontsize=10, fontname="helv")

    doc.save(str(file_path))
    doc.close()


def create_pyq_pdf(
    file_path: Path,
    subject_code: str,
    subject_name: str,
    session: str,
    year: int,
    questions: list[dict],
    include_scanned_page: bool = False,
):
    doc = pymupdf.open()
    # Page 1: Header + Module 1, 2, 3
    page1 = doc.new_page(width=595, height=842)
    header = f"""USN: ____________________
VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI
Fifth Semester B.E. Degree Examination, {session} ({year})
Subject: {subject_name}
Subject Code: {subject_code}
Time: 3 Hours                                                    Max. Marks: 100

Note: Answer any FIVE full questions, choosing ONE full question from each module.
----------------------------------------------------------------------------------------------------
"""
    y = 50
    page1.insert_text((40, y), header, fontsize=10, fontname="helv")
    y += 110

    p1_questions = [q for q in questions if q.get("module") in (1, 2, 3)]
    for q in p1_questions:
        q_text = f"Module {q['module']} | Q.{q['q_no']} ({q['marks']} Marks)\n{q['text']}\n"
        page1.insert_text((40, y), q_text, fontsize=9.5, fontname="helv")
        y += 75

    # Page 2: Module 4, 5
    if include_scanned_page:
        # Create an image-only page (fewer than 50 text characters in pdf stream) to trigger OCR fallback
        img = Image.new("RGB", (1200, 1600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw simulated scanned text
        scanned_content = f"""VISVESVARAYA TECHNOLOGICAL UNIVERSITY
{subject_code} - {subject_name} | {session}
Module 4 & Module 5 (Scanned Archival Section)

Module 4 | Q.7a (10 Marks)
Explain COCOMO II estimation model and project scheduling fundamentals.

Module 4 | Q.8a (10 Marks)
Describe risk management stages: identification, analysis, planning, and monitoring.

Module 5 | Q.9a (10 Marks)
Explain software quality assurance and standard verification techniques.

Module 5 | Q.10a (10 Marks)
Describe software evolution dynamics, reverse engineering, and legacy maintenance.
"""
        draw.text((80, 100), scanned_content, fill=(0, 0, 0))
        img_path = file_path.parent / f"temp_scan_{file_path.stem}.png"
        img.save(str(img_path))

        page2 = doc.new_page(width=595, height=842)
        # Insert image without text stream
        page2.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(img_path))
        if img_path.exists():
            img_path.unlink()
    else:
        page2 = doc.new_page(width=595, height=842)
        y = 50
        page2.insert_text(
            (40, y),
            f"{subject_code} - {subject_name} | {session} (Page 2)\n----------------------------------------------------------------------------------------------------\n",
            fontsize=10,
            fontname="helv",
        )
        y += 40
        p2_questions = [q for q in questions if q.get("module") in (4, 5)]
        for q in p2_questions:
            q_text = f"Module {q['module']} | Q.{q['q_no']} ({q['marks']} Marks)\n{q['text']}\n"
            page2.insert_text((40, y), q_text, fontsize=9.5, fontname="helv")
            y += 75

    doc.save(str(file_path))
    doc.close()


def generate_all_raw_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. 21CS51 - Software Engineering & Project Management
    modules_21cs51 = {
        "module_1": "Introduction to Software Engineering: Professional software development, Software engineering ethics, Case studies. Software Process Models: Waterfall model, Incremental development, Integration and configuration. Agile Software Development: Agile methods, Agile development techniques, Extreme Programming (XP), Scrum framework.",
        "module_2": "Requirements Engineering: Functional and non-functional requirements, Requirements engineering processes, Requirements elicitation, Requirements specification, Requirements validation. System Modeling: Context models, Interaction models (Use Case and Sequence diagrams), Structural models (Class diagrams), Behavioral models (State machine diagrams).",
        "module_3": "Architectural Design: Architectural design decisions, Architectural views, Architectural patterns (Model-View-Controller, Layered architecture, Repository pattern, Client-Server). Design and Implementation: Object-oriented design using UML, Design patterns, Open source development. Software Testing: Development testing, Unit testing, Component testing, System testing, Acceptance testing, Test-driven development (TDD).",
        "module_4": "Software Evolution: Evolution processes, Program evolution dynamics, Software maintenance, Re-engineering, Reverse engineering. Project Planning: Software pricing, Plan-driven development, Project scheduling, Agile planning. Estimation Techniques: Algorithmic cost modeling, COCOMO II model, Function point analysis.",
        "module_5": "Project Management: Risk management, Risk identification, Risk analysis, Risk planning, Risk monitoring. Quality Management: Software quality, Software standards, Quality planning, Quality assurance and control, Software metrics for quality assessment.",
    }
    tb_21cs51 = [
        "Ian Sommerville, 'Software Engineering', 10th Edition, Pearson Education, 2017.",
        "Roger S. Pressman, 'Software Engineering: A Practitioner's Approach', 8th Edition, McGraw-Hill, 2015.",
    ]
    create_syllabus_pdf(
        data_dir / "2022_CSE_5_21CS51_syl.pdf",
        "21CS51",
        "Software Engineering & Project Management",
        3,
        modules_21cs51,
        tb_21cs51,
    )

    # 2. 21CS52 - Computer Networks
    modules_21cs52 = {
        "module_1": "Computer Networks and the Internet: What is the Internet? The Network Edge (end systems, access networks), The Network Core (packet switching, circuit switching, network of networks), Delay, Loss, and Throughput in Packet-Switched Networks, Protocol Layers and Their Service Models (TCP/IP and OSI reference models).",
        "module_2": "Application Layer: Principles of Network Applications, The Web and HTTP (Persistent and Non-Persistent Connections, Cookies, Web Caching), Electronic Mail (SMTP, POP3, IMAP), DNS: The Internet's Directory Service (DNS services, hierarchy, DNS records), Socket Programming with UDP and TCP.",
        "module_3": "Transport Layer: Transport-Layer Services, Multiplexing and Demultiplexing, Connectionless Transport: UDP, Principles of Reliable Data Transfer (Stop-and-Wait, Go-Back-N, Selective Repeat protocols), Connection-Oriented Transport: TCP (Segment structure, Round-trip time estimation, Reliable data transfer, Flow control, Connection management, TCP Congestion Control).",
        "module_4": "The Network Layer - Data Plane & Control Plane: Overview of Network Layer, Router Architecture, The Internet Protocol (IP): IPv4 addressing, Subnetting, Classless Interdomain Routing (CIDR), DHCP, NAT. Routing Algorithms: Link-State (Dijkstra's Algorithm), Distance-Vector (Bellman-Ford Algorithm). Intra-AS Routing: OSPF, Inter-AS Routing: BGP.",
        "module_5": "The Link Layer and LANs: Link Layer Services, Error-Detection and Correction Techniques (Parity checks, Checksumming, Cyclic Redundancy Check CRC), Multiple Access Links and Protocols (Random access: ALOHA, CSMA/CD, CSMA/CA), Switched Local Area Networks: Link-layer addressing and ARP, Ethernet, Link-layer switches, Virtual Local Area Networks (VLANs).",
    }
    tb_21cs52 = [
        "James F. Kurose and Keith W. Ross, 'Computer Networking: A Top-Down Approach', 8th Edition, Pearson, 2021.",
        "Andrew S. Tanenbaum, David J. Wetherall, 'Computer Networks', 5th Edition, Pearson, 2013.",
    ]
    create_syllabus_pdf(
        data_dir / "2022_CSE_5_21CS52_syl.pdf",
        "21CS52",
        "Computer Networks",
        4,
        modules_21cs52,
        tb_21cs52,
    )

    # 3. 21CS53 - Database Management Systems
    modules_21cs53 = {
        "module_1": "Databases and Database Users: Introduction, Characteristics of the Database Approach, Advantages of using DBMS approach. Database System Concepts and Architecture: Data Models, Schemas, and Instances, Three-Schema Architecture and Data Independence, Database Languages and Interfaces. Data Modeling using Entity-Relationship (ER) Model: Entity types, Entity Sets, Attributes, Relationships, Constraints, Enhanced ER (EER) modeling.",
        "module_2": "The Relational Data Model and Relational Database Constraints: Relational model concepts, Relational integrity constraints. Relational Algebra: Unary relational operations (SELECT, PROJECT), Relational algebra operations from set theory (UNION, INTERSECT, DIFFERENCE, CARTESIAN PRODUCT), Binary operations (JOIN, DIVISION). SQL: Schema definition, Constraints, Queries, Nested subqueries, Aggregation operators, Views.",
        "module_3": "Database Design Theory and Normalization: Informal design guidelines for relation schemas, Functional Dependencies, Normal Forms based on primary keys: First Normal Form (1NF), Second Normal Form (2NF), Third Normal Form (3NF), Boyce-Codd Normal Form (BCNF), Multi-valued Dependencies and Fourth Normal Form (4NF).",
        "module_4": "Transaction Processing: Introduction to Transaction Processing, Transaction and System Concepts, Desirable Properties of Transactions (ACID properties), Characterizing Schedules based on Recoverability and Serializability. Concurrency Control Techniques: Two-Phase Locking (2PL) techniques, Concurrency control based on Timestamp Ordering, Multiversion concurrency control, Deadlock handling.",
        "module_5": "Database Recovery Techniques: Recovery concepts, NO-UNDO/REDO recovery based on deferred update, Recovery techniques based on immediate update (UNDO/REDO), Shadow Paging, Checkpoints. Indexing Structures for Files: Primary, Clustering, and Secondary Indexes, Multilevel Indexes, Dynamic Multilevel Indexes using B-Trees and B+ Trees.",
    }
    tb_21cs53 = [
        "Ramez Elmasri and Shamkant B. Navathe, 'Fundamentals of Database Systems', 7th Edition, Pearson, 2017.",
        "Abraham Silberschatz, Henry F. Korth, S. Sudarshan, 'Database System Concepts', 7th Edition, McGraw-Hill, 2019.",
    ]
    create_syllabus_pdf(
        data_dir / "2022_CSE_5_21CS53_syl.pdf",
        "21CS53",
        "Database Management Systems",
        4,
        modules_21cs53,
        tb_21cs53,
    )

    # 4. 21CS54 - Artificial Intelligence & Machine Learning
    modules_21cs54 = {
        "module_1": "Introduction to AI: What is Artificial Intelligence? Problems, State space search, Production systems, Problem characteristics. Heuristic Search Techniques: Generate-and-Test, Hill Climbing, Simulated Annealing, Best-First Search, A* Algorithm, AO* Algorithm, Problem Reduction.",
        "module_2": "Knowledge Representation: Representations and Mappings, Approaches to Knowledge Representation, Issues in Knowledge Representation. Using Predicate Logic: Representing simple facts in logic, Representing instance and isa relationships, Computable functions and predicates, Resolution in Propositional and First-Order Predicate Logic, Unification algorithm.",
        "module_3": "Game Playing: The Minimax Search Procedure, Adding Alpha-Beta Cutoffs, Additional Refinements. Planning: Overview of Planning, An Example Domain: The Blocks World, Components of a Planning System, Goal Stack Planning, Nonlinear Planning using Constraint Posting.",
        "module_4": "Introduction to Machine Learning: Well-posed learning problems, Designing a learning system, Perspectives and issues in machine learning. Concept Learning: Concept learning task, General-to-Specific ordering, Find-S algorithm, Version Spaces and Candidate Elimination algorithm. Decision Tree Learning: Decision tree representation, ID3 algorithm, Hypothesis space search, Inductive bias, Issues in decision tree learning.",
        "module_5": "Artificial Neural Networks: Neural network representation, Perceptrons, Training a perceptron, Multilayer networks and the Backpropagation algorithm. Unsupervised Learning: Clustering, K-Means clustering algorithm, Hierarchical clustering, Dimensionality reduction: Principal Component Analysis (PCA).",
    }
    tb_21cs54 = [
        "Elaine Rich, Kevin Knight, Shivashankar B Nair, 'Artificial Intelligence', 3rd Edition, McGraw-Hill, 2017.",
        "Tom M. Mitchell, 'Machine Learning', McGraw-Hill, 1997.",
    ]
    create_syllabus_pdf(
        data_dir / "2022_CSE_5_21CS54_syl.pdf",
        "21CS54",
        "Artificial Intelligence & Machine Learning",
        3,
        modules_21cs54,
        tb_21cs54,
    )

    # Now PYQs for 21CS51 (Jan 2023 - with scanned page for OCR test, and July 2024)
    q_21cs51_2023 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Explain Waterfall software process model with a neat diagram. Discuss its advantages and limitations."},
        {"module": 1, "q_no": "2a", "marks": 10, "text": "Describe the Scrum agile framework. Explain the roles of Product Owner, Scrum Master, and Sprint execution."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "What is requirements elicitation? Differentiate between functional and non-functional requirements with examples."},
        {"module": 2, "q_no": "4a", "marks": 10, "text": "Explain Use Case diagrams and Sequence diagrams in system modeling with a banking system example."},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Describe Model-View-Controller (MVC) architectural pattern. Draw its structure and discuss its advantages."},
        {"module": 3, "q_no": "6a", "marks": 10, "text": "Differentiate between Unit testing, Component testing, System testing, and Acceptance testing."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS51_pyq_2023.pdf",
        "21CS51",
        "Software Engineering & Project Management",
        "Dec 2022 / Jan 2023",
        2023,
        q_21cs51_2023,
        include_scanned_page=True,  # triggers OCR fallback!
    )

    q_21cs51_2024 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Discuss Extreme Programming (XP) practices including pair programming, test-first development, and continuous integration."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Explain requirements validation techniques: requirements reviews, prototyping, and test-case generation."},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Explain Layered architecture pattern and Repository pattern. When should each pattern be used?"},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain Algorithmic cost modeling and COCOMO II cost estimation sub-models with formulae."},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain risk management process: risk identification, risk analysis, risk planning, and risk monitoring."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS51_pyq_2024.pdf",
        "21CS51",
        "Software Engineering & Project Management",
        "June / July 2024",
        2024,
        q_21cs51_2024,
    )

    # PYQs for 21CS52 (Computer Networks)
    q_21cs52_2023 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Compare Packet Switching and Circuit Switching. Explain transmission delay, propagation delay, queuing delay, and processing delay."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Explain HTTP persistent and non-persistent connections with timing diagrams. What is the role of Web Caching?"},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Explain the Go-Back-N and Selective Repeat sliding window protocols. How do they differ in buffering and retransmission?"},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain Dijkstra's Link-State routing algorithm. Apply it to find the shortest paths from source node u in a given graph."},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain Cyclic Redundancy Check (CRC) error detection technique with an illustrative numerical example."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS52_pyq_2023.pdf",
        "21CS52",
        "Computer Networks",
        "June / July 2023",
        2023,
        q_21cs52_2023,
    )

    q_21cs52_2024 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Explain OSI reference model and TCP/IP protocol suite. Discuss the responsibilities of each layer."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Explain DNS architecture, DNS message format, and hierarchy of DNS servers (Root, TLD, Authoritative)."},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Explain TCP 3-way handshake connection establishment, TCP segment structure, and TCP congestion control (Slow Start, Congestion Avoidance, Fast Recovery)."},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain Distance Vector routing algorithm based on Bellman-Ford equation. What is the Count-to-Infinity problem and how is Split Horizon used to mitigate it?"},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain CSMA/CD protocol used in Ethernet. How does exponential backoff algorithm work upon collision?"},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS52_pyq_2024.pdf",
        "21CS52",
        "Computer Networks",
        "Dec 2023 / Jan 2024",
        2024,
        q_21cs52_2024,
    )

    # PYQs for 21CS53 (DBMS)
    q_21cs53_2023 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Explain Three-Schema Architecture of DBMS. Distinguish between logical and physical data independence."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Explain fundamental relational algebra operations: SELECT, PROJECT, UNION, SET DIFFERENCE, CARTESIAN PRODUCT with syntax and examples."},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Define Functional Dependency. Explain First Normal Form (1NF), Second Normal Form (2NF), and Third Normal Form (3NF) with relation examples."},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain ACID properties of database transactions. What is conflict serializability and how is precedence graph used to test conflict serializability?"},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain Write-Ahead Logging (WAL) and Checkpoint mechanisms in database recovery."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS53_pyq_2023.pdf",
        "21CS53",
        "Database Management Systems",
        "Dec 2022 / Jan 2023",
        2023,
        q_21cs53_2023,
    )

    q_21cs53_2024 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Construct an ER diagram for a University Management System specifying entity sets, relationship types, cardinalities, and primary keys."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Write SQL queries for nested queries, aggregate functions (COUNT, SUM, AVG, GROUP BY, HAVING), and Views on a Company database schema."},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Explain Boyce-Codd Normal Form (BCNF). How does BCNF differ from 3NF? Give an example of a relation in 3NF but not in BCNF."},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain Two-Phase Locking (2PL) protocol: Basic 2PL, Conservative 2PL, and Strict 2PL. How does strict 2PL prevent cascading aborts?"},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain B-Trees and B+ Trees indexing structures. Contrast search, insertion, and deletion efficiency in B+ Trees versus B-Trees."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS53_pyq_2024.pdf",
        "21CS53",
        "Database Management Systems",
        "Dec 2023 / Jan 2024",
        2024,
        q_21cs53_2024,
    )

    # PYQs for 21CS54 (AIML)
    q_21cs54_2023 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Explain A* algorithm with its pseudo-code. Prove that A* is admissible when heuristic h(n) is an underestimate."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Explain Resolution algorithm in Propositional Logic and First-Order Predicate Logic. What is Unification algorithm?"},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Explain Minimax search procedure and Alpha-Beta Pruning with a game tree example showing pruned branches."},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain ID3 Decision Tree algorithm. Define Entropy and Information Gain used for attribute selection."},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain K-Means clustering algorithm step by step. Discuss how to choose the value of k using the Elbow method."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS54_pyq_2023.pdf",
        "21CS54",
        "Artificial Intelligence & Machine Learning",
        "June / July 2023",
        2023,
        q_21cs54_2023,
    )

    q_21cs54_2024 = [
        {"module": 1, "q_no": "1a", "marks": 10, "text": "Explain State Space Search representation. Discuss Hill Climbing algorithm and its limitations (Local Maxima, Ridges, Plateaus)."},
        {"module": 2, "q_no": "3a", "marks": 10, "text": "Explain Semantic Networks and Frames representation schemes with appropriate knowledge representation examples."},
        {"module": 3, "q_no": "5a", "marks": 10, "text": "Explain Goal Stack Planning in the Blocks World domain with an example demonstrating start state, goal state, and operator stack."},
        {"module": 4, "q_no": "7a", "marks": 10, "text": "Explain Candidate Elimination algorithm and Version Spaces with an example training dataset."},
        {"module": 5, "q_no": "9a", "marks": 10, "text": "Explain Multilayer Perceptron and Backpropagation algorithm for training Artificial Neural Networks."},
    ]
    create_pyq_pdf(
        data_dir / "2022_CSE_5_21CS54_pyq_2024.pdf",
        "21CS54",
        "Artificial Intelligence & Machine Learning",
        "June / July 2024",
        2024,
        q_21cs54_2024,
    )


if __name__ == "__main__":
    out_dir = Path("data/raw")
    generate_all_raw_data(out_dir)
    print(f"Generated raw VTU PDFs in {out_dir.resolve()}:")
    for p in sorted(out_dir.glob("*.pdf")):
        print(f" - {p.name}")
