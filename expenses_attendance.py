"""Expense, Attendance, Grading and Billing Manager.
"""

import math
from datetime import date


TAX_RATE = 0.18
PASS_MARK = 35
ELIGIBLE_LIMIT = 75
WARNING_LIMIT = 60
DEFAULT_SUBJECTS = 5
STOP_WORD = "done"

CATEGORIES = ("Food", "Travel", "Utilities", "Stationery", "Other")
DEFAULT_SUBJECT_NAMES = ("Maths", "Science", "English", "History", "Computer")

# (minimum subtotal, discount rate) - biggest slab first
DISCOUNT_SLABS = ((10000, 0.15), (5000, 0.10), (1000, 0.05), (0, 0.0))

# (minimum percentage, grade, remark)
GRADE_SCALE = (
    (90, "A+", "Outstanding"),
    (80, "A", "Excellent"),
    (70, "B", "Very Good"),
    (60, "C", "Good"),
    (50, "D", "Satisfactory"),
    (35, "E", "Pass"),
    (0, "F", "Fail"),
)

# ---------- data kept in memory ----------
expenses = []        # each: {"description", "category", "amount"}
people = []          # each: {"name", "id", "present"}
students = []        # each: {"name", "marks", "total", ...}
bills = []           # each: finished bill dicts
settings = {
    "total_days": None,
    "subjects": list(DEFAULT_SUBJECT_NAMES[:DEFAULT_SUBJECTS]),
}


# =====================================================
# input helpers
# =====================================================

def get_int(prompt, min_val=0, max_val=None, error=None):

    if error is None:
        error = "Enter a whole number."
    while True:
        text = input(prompt).strip()
        try:
            value = int(text)
        except ValueError:
            print(error)
            continue
        if value < min_val or (max_val is not None and value > max_val):
            print(error)
            continue
        return value


def get_float(prompt, min_val=0.0, max_val=float("inf"), error=None):

    if error is None:
        error = f"Value must be between {min_val} and {max_val}."
    while True:
        text = input(prompt).strip()
        try:
            value = float(text)
        except ValueError:
            print(error)
            continue
        # float("nan") and float("inf") parse fine, so check them here
        if not math.isfinite(value) or value < min_val or value > max_val:
            print(error)
            continue
        return value


def get_name(prompt):

    while True:
        name = " ".join(input(prompt).split())
        letters = sum(1 for ch in name if ch.isalpha())
        only_valid = all(ch.isalpha() or ch == " " for ch in name)
        if only_valid and letters >= 2 and len(name) <= 40:
            return name
        print("Name must contain 2-40 letters.")


def get_choice(prompt, options):

    lookup = {opt.lower(): opt for opt in options}
    while True:
        text = input(prompt).strip().lower()
        if text in lookup:
            return lookup[text]
        print("Choose a category from the list.")


def get_text(prompt, error, max_len=40):

    while True:
        text = " ".join(input(prompt).split())
        if text and len(text) <= max_len:
            return text
        print(error)


def get_yes_no(prompt):
    while True:
        text = input(prompt).strip().lower()
        if text in ("y", "yes"):
            return True
        if text in ("n", "no"):
            return False
        print("Please type yes or no.")


def get_menu_choice(low, high):
    return get_int(f"Enter your choice ({low}-{high}): ", low, high,
                   f"Invalid choice. Enter {low}-{high}.")



# expenses


def add_expense():
    description = get_text("Description: ", "Description cannot be empty.")
    category = get_choice("Category (" + "/".join(CATEGORIES) + "): ", CATEGORIES)
    amount = get_float("Amount: ", 0.01, float("inf"),
                       "Amount must be a positive number.")
    expenses.append({
        "description": description,
        "category": category,
        "amount": round(amount, 2),
    })
    print("Expense added successfully!")


def category_totals():
    totals = {}
    for item in expenses:
        totals[item["category"]] = totals.get(item["category"], 0) + item["amount"]
    return totals


def expense_list():
    if not expenses:
        return None
    lines = ["=== ALL EXPENSES ===",
             f"{'No':<4}{'Description':<25}{'Category':<12}{'Amount':>12}",
             "-" * 53]
    for number, item in enumerate(expenses, start=1):
        lines.append(f"{number:<4}{item['description']:<25}{item['category']:<12}"
                     f"{item['amount']:>12.2f}")
    return "\n".join(lines)


def expense_summary():
    if not expenses:
        return None
    totals = category_totals()
    lines = ["=== EXPENSE SUMMARY ===",
             f"{'CATEGORY':<15}{'TOTAL':>12}"]
    for category in CATEGORIES:
        if category in totals:
            lines.append(f"{category:<15}{totals[category]:>12.2f}")
    lines.append("-" * 27)
    grand = round(sum(item["amount"] for item in expenses), 2)
    lines.append(f"{'GRAND TOTAL':<15}{grand:>12.2f}")
    highest = max(expenses, key=lambda e: e["amount"])
    lowest = min(expenses, key=lambda e: e["amount"])
    lines.append(f"Highest expense: {highest['description']} "
                 f"({highest['category']}) - {highest['amount']:.2f}")
    lines.append(f"Lowest expense : {lowest['description']} "
                 f"({lowest['category']}) - {lowest['amount']:.2f}")
    return "\n".join(lines)


def expense_report():
    if not expenses:
        return None
    return expense_list() + "\n\n" + expense_summary()


def expense_menu():
    while True:
        print("\n--- EXPENSE MANAGEMENT ---")
        print(" 1. Add Expense")
        print(" 2. View All Expenses")
        print(" 3. Expense Summary")
        print(" 4. Back")
        choice = get_menu_choice(1, 4)
        if choice == 1:
            add_expense()
        elif choice == 2:
            show(expense_list())
        elif choice == 3:
            show(expense_summary())
        else:
            return



# attendance


def attendance_percentage(present, total_days):
    if total_days <= 0:
        return 0.0
    return round(present / total_days * 100, 2)


def attendance_status(percentage):
    if percentage >= ELIGIBLE_LIMIT:
        return "Eligible"
    elif percentage >= WARNING_LIMIT:
        return "Warning"
    else:
        return "Shortage"


def set_total_days():
    
    lowest_allowed = max([p["present"] for p in people], default=0)
    lowest_allowed = max(lowest_allowed, 1)
    settings["total_days"] = get_int(
        "Total working days: ", lowest_allowed, None,
        "Enter a whole number of at least 1." if lowest_allowed == 1
        else f"Enter a whole number of at least {lowest_allowed}.")
    print("Working days saved.")


def mark_attendance():
    if settings["total_days"] is None:
        print("Set the total working days first.")
        set_total_days()
    total = settings["total_days"]
    name = get_name("Name: ")
    while True:
        person_id = get_text("ID: ", "ID cannot be empty (max 40 characters).", 10)
        if not person_id.isalnum():
            print("ID can only have letters and numbers.")
        elif any(p["id"].lower() == person_id.lower() for p in people):
            print("That ID is already recorded.")
        else:
            break
    present = get_int(f"Days present (0-{total}): ", 0, total,
                      f"Days present must be between 0 and {total}.")
    people.append({"name": name, "id": person_id, "present": present})
    print("Attendance recorded.")


def attendance_table():
    if not people:
        return None
    total = settings["total_days"]
    lines = ["=== ATTENDANCE ===",
             f"{'Name':<22}{'ID':<10}{'Total Days':>11}{'Present':>9}"
             f"{'Attendance %':>14}  Status",
             "-" * 77]
    for p in people:
        pct = attendance_percentage(p["present"], total)
        lines.append(f"{p['name']:<22}{p['id']:<10}{total:>11}{p['present']:>9}"
                     f"{pct:>14.2f}  {attendance_status(pct)}")
    return "\n".join(lines)


def attendance_summary():
    if not people:
        return None
    total = settings["total_days"]
    percentages = [attendance_percentage(p["present"], total) for p in people]
    average = sum(percentages) / len(percentages)
    best = percentages.index(max(percentages))
    worst = percentages.index(min(percentages))
    counts = {"Eligible": 0, "Warning": 0, "Shortage": 0}
    for pct in percentages:
        counts[attendance_status(pct)] += 1
    return "\n".join([
        "Summary:",
        f"Total students     : {len(people)}",
        f"Average attendance : {average:.2f}%",
        f"Highest            : {people[best]['name']} ({percentages[best]:.2f}%)",
        f"Lowest             : {people[worst]['name']} ({percentages[worst]:.2f}%)",
        f"Eligible: {counts['Eligible']} | Warning: {counts['Warning']} "
        f"| Shortage: {counts['Shortage']}",
    ])


def attendance_report():
    if not people:
        return None
    return attendance_table() + "\n\n" + attendance_summary()


def attendance_menu():
    while True:
        print("\n--- ATTENDANCE MANAGEMENT ---")
        print(" 1. Set Total Working Days")
        print(" 2. Add Person's Attendance")
        print(" 3. View Attendance and Summary")
        print(" 4. Back")
        choice = get_menu_choice(1, 4)
        if choice == 1:
            set_total_days()
        elif choice == 2:
            mark_attendance()
        elif choice == 3:
            show(attendance_report())
        else:
            return



# grading


def calculate_grade(percentage):
    for minimum, grade, remark in GRADE_SCALE:
        if percentage >= minimum:
            return grade, remark
    return "F", "Fail"


def fmt_marks(value):
    return f"{round(value, 2):g}"


def set_subject_count():
    if students:
        print("Subjects can't be changed after students have been entered.")
        return
    count = get_int("Number of subjects (1-15): ", 1, 15,
                    "Enter a whole number between 1 and 15.")
    names = []
    for i in range(count):
        if i < len(DEFAULT_SUBJECT_NAMES):
            default = DEFAULT_SUBJECT_NAMES[i]
        else:
            default = f"Subject {i + 1}"
        typed = input(f"Name of subject {i + 1} [{default}]: ").strip()
        names.append(typed if typed else default)
    settings["subjects"] = names
    print(f"{count} subjects saved.")


def add_student():
    name = get_name("Student name: ")
    marks = []
    for subject in settings["subjects"]:
        marks.append(get_float(f"Marks in {subject} (0-100): ", 0, 100,
                               "Marks must be between 0 and 100."))
    total = sum(marks)
    maximum = 100 * len(marks)
    average = total / len(marks)
    percentage = round(total / maximum * 100, 2)
    grade, remark = calculate_grade(percentage)
    failed = any(m < PASS_MARK for m in marks)
    students.append({
        "name": name,
        "marks": marks,
        "total": total,
        "maximum": maximum,
        "average": round(average, 2),
        "percentage": percentage,
        "grade": grade,
        "result": "FAIL" if failed else "PASS",
    })
    print("Student added.")
    print(generate_report_card(students[-1]))


def generate_report_card(student):
    lines = ["=========== REPORT CARD ===========",
             f"Student : {student['name']}",
             "-" * 35,
             f"{'Subject':<15}{'Marks':>8}"]
    for subject, mark in zip(settings["subjects"], student["marks"]):
        lines.append(f"{subject:<15}{fmt_marks(mark):>8}")
    lines.append("-" * 35)
    lines.append(f"Total      : {fmt_marks(student['total'])} / {student['maximum']}")
    lines.append(f"Average    : {student['average']:.2f}")
    lines.append(f"Percentage : {student['percentage']:.2f}%")
    lines.append(f"Grade      : {student['grade']}")
    lines.append(f"Result     : {student['result']}")
    lines.append("=" * 35)
    return "\n".join(lines)


def class_summary():
    if not students:
        return None
    count = len(students)
    class_avg = sum(s["percentage"] for s in students) / count
    topper = max(students, key=lambda s: s["percentage"])
    passed = sum(1 for s in students if s["result"] == "PASS")
    return "\n".join([
        "=== CLASS SUMMARY ===",
        f"{'Name':<22}{'Percentage':>11}{'Grade':>7}  Result",
        "-" * 50,
        *[f"{s['name']:<22}{s['percentage']:>11.2f}{s['grade']:>7}  {s['result']}"
          for s in students],
        "-" * 50,
        f"Students      : {count}",
        f"Class average : {class_avg:.2f}%",
        f"Topper        : {topper['name']} ({topper['percentage']:.2f}%)",
        f"Passed: {passed} | Failed: {count - passed}",
    ])


def grading_report():
    if not students:
        return None
    cards = [generate_report_card(s) for s in students]
    return "\n\n".join(cards) + "\n\n" + class_summary()


def grading_menu():
    while True:
        print("\n--- GRADING SYSTEM ---")
        print(" 1. Add Student Marks")
        print(" 2. View Class Summary")
        print(" 3. Change Subjects")
        print(" 4. Back")
        choice = get_menu_choice(1, 4)
        if choice == 1:
            add_student()
        elif choice == 2:
            show(class_summary())
        elif choice == 3:
            set_subject_count()
        else:
            return



# billing


def calculate_discount(subtotal):
    for minimum, rate in DISCOUNT_SLABS:
        if subtotal >= minimum:
            return rate
    return 0.0


def calculate_tax(taxable):
    return round(taxable * TAX_RATE, 2)


def create_bill():
    customer = get_name("Customer name: ")
    bill_no = f"B-{len(bills) + 1:04d}"
    print(f"Bill number: {bill_no}")
    print(f"Add items one by one. Type '{STOP_WORD}' as the item name to finish.")

    items = []
    while True:
        name = get_text("Item name: ", "Item name cannot be empty.")
        if name.lower() == STOP_WORD:
            break
        price = get_float("Unit price: ", 0.01, float("inf"),
                          "Price must be greater than 0.")
        quantity = get_int("Quantity: ", 1, None,
                           "Quantity must be a whole number >= 1.")
        price = round(price, 2)
        items.append({
            "name": name,
            "price": price,
            "qty": quantity,
            "total": round(price * quantity, 2),
        })

    if not items:
        print("No items added. Bill cancelled.")
        return

    subtotal = round(sum(i["total"] for i in items), 2)
    rate = calculate_discount(subtotal)
    discount_amt = round(subtotal * rate, 2)
    taxable = round(subtotal - discount_amt, 2)
    tax_amt = calculate_tax(taxable)
    grand_total = round(taxable + tax_amt, 2)

    bill = {
        "bill_no": bill_no,
        "date": date.today().isoformat(),
        "customer": customer,
        "items": items,
        "subtotal": subtotal,
        "rate": rate,
        "discount": discount_amt,
        "taxable": taxable,
        "tax": tax_amt,
        "grand_total": grand_total,
    }
    bills.append(bill)
    print()
    print(generate_invoice(bill))


def generate_invoice(bill):
    lines = ["=============== INVOICE ===============",
             f"Bill No : {bill['bill_no']}    Date: {bill['date']}",
             f"Customer : {bill['customer']}",
             "-" * 39,
             f"{'Item':<15}{'Price':>8}{'Qty':>5}{'Total':>11}"]
    for item in bill["items"]:
        lines.append(f"{item['name'][:14]:<15}{item['price']:>8.2f}"
                     f"{item['qty']:>5}{item['total']:>11.2f}")
    lines.append("-" * 39)
    lines.append(f"{'Subtotal':<24}{bill['subtotal']:>15.2f}")
    lines.append(f"{'Discount (' + format(bill['rate'] * 100, 'g') + '%)':<24}"
                 f"{-bill['discount']:>15.2f}")
    lines.append(f"{'Taxable amount':<24}{bill['taxable']:>15.2f}")
    lines.append(f"{'Tax (' + format(TAX_RATE * 100, 'g') + '%)':<24}"
                 f"{bill['tax']:>15.2f}")
    lines.append("-" * 39)
    lines.append(f"{'GRAND TOTAL':<24}{bill['grand_total']:>15.2f}")
    lines.append("=" * 39)
    lines.append("Thank you for your purchase!")
    return "\n".join(lines)


def billing_report():
    if not bills:
        return None
    lines = ["=== BILLING REPORT ===",
             f"{'Bill No':<10}{'Date':<12}{'Customer':<22}{'Total':>12}",
             "-" * 56]
    for b in bills:
        lines.append(f"{b['bill_no']:<10}{b['date']:<12}{b['customer']:<22}"
                     f"{b['grand_total']:>12.2f}")
    lines.append("-" * 56)
    sales = round(sum(b["grand_total"] for b in bills), 2)
    lines.append(f"{'Bills: ' + str(len(bills)):<44}{sales:>12.2f}")
    return "\n".join(lines)


def billing_menu():
    while True:
        print("\n--- BILLING SYSTEM ---")
        print(" 1. Create New Bill")
        print(" 2. View All Bills")
        print(" 3. Back")
        choice = get_menu_choice(1, 3)
        if choice == 1:
            create_bill()
        elif choice == 2:
            show(billing_report())
        else:
            return



# reports


def show(text):
    """Print a report, or a message if there is nothing to show."""
    print()
    if text is None:
        print("No records available.")
    else:
        print(text)


def export_report(text, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Report saved to {filename}")
    except OSError as err:
        print(f"Could not save the file: {err}")


def reports_menu():
    reports = {
        1: ("expense", expense_report),
        2: ("attendance", attendance_report),
        3: ("grading", grading_report),
        4: ("billing", billing_report),
    }
    while True:
        print("\n--- REPORTS ---")
        print(" 1. Expense Report")
        print(" 2. Attendance Report")
        print(" 3. Grading Report")
        print(" 4. Billing Report")
        print(" 5. Back")
        choice = get_menu_choice(1, 5)
        if choice == 5:
            return
        name, builder = reports[choice]
        text = builder()
        show(text)
        if text is not None and get_yes_no("\nSave this report to a .txt file? (yes/no): "):
            export_report(text, f"{name}_report.txt")


# =====================================================
# main menu
# =====================================================

def display_menu():
    print()
    print("=" * 46)
    print("  EXPENSE | ATTENDANCE | GRADING | BILLING")
    print("=" * 46)
    print(" 1. Expense Management")
    print(" 2. Attendance Management")
    print(" 3. Grading System")
    print(" 4. Billing System")
    print(" 5. View Reports")
    print(" 6. Exit")
    print("-" * 46)


def main():
    modules = {
        1: expense_menu,
        2: attendance_menu,
        3: grading_menu,
        4: billing_menu,
        5: reports_menu,
    }
    try:
        while True:
            display_menu()
            choice = get_menu_choice(1, 6)
            if choice == 6:
                print("Thank you for using the system. Goodbye!")
                break
            modules[choice]()
    except (KeyboardInterrupt, EOFError):
        print("\n\nInput stopped. Goodbye!")


if __name__ == "__main__":
    main()
