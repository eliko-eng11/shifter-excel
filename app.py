import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# =========================
# חובה: page_config ראשון
# =========================
st.set_page_config(page_title="מערכת שיבוץ משמרות (Excel)", layout="wide")


# =========================
# הקצאה חמדנית (ללא scipy)
# =========================
def simple_assignment(cost_matrix):
    used_rows = set()
    used_cols = set()
    assignments = []

    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows > 0 else 0

    for _ in range(min(rows, cols)):
        best = None
        best_cost = 10**18

        for i in range(rows):
            if i in used_rows:
                continue
            for j in range(cols):
                if j in used_cols:
                    continue
                c = cost_matrix[i][j]
                if c < best_cost:
                    best_cost = c
                    best = (i, j)

        if best is None:
            break

        r, c = best
        assignments.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    if not assignments:
        return [], []

    rr, cc = zip(*assignments)
    return list(rr), list(cc)


# =========================
# בניית שיבוץ מתוך 3 טאבים
# =========================
def build_schedule(workers_df, req_df, pref_df, week_number: int):
    # ניקוי שמות עמודות
    workers_df.columns = workers_df.columns.str.strip()
    req_df.columns = req_df.columns.str.strip()
    pref_df.columns = pref_df.columns.str.strip()

    # התאמת שמות עמודות בעברית לאנגלית פנימית (וגם תומך באנגלית)
    workers_df = workers_df.rename(columns={"שם עובד": "worker"})
    req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df = pref_df.rename(columns={"עדיפות": "preference", "עובד": "worker", "יום": "day", "משמרת": "shift"})

    # בדיקות כותרות
    if "worker" not in workers_df.columns:
        raise ValueError("ב-workers חייבת להיות עמודה בשם 'worker' (או 'שם עובד').")
    for c in ["day", "shift", "required"]:
        if c not in req_df.columns:
            raise ValueError("ב-requirements חייבות להיות עמודות: day, shift, required (או בעברית יום/משמרת/כמות נדרשת).")
    for c in ["worker", "day", "shift", "preference"]:
        if c not in pref_df.columns:
            raise ValueError("ב-preferences חייבות להיות עמודות: worker, day, shift, preference (או בעברית עובד/יום/משמרת/עדיפות).")

    # ניקוי רווחים
    workers_df["worker"] = workers_df["worker"].astype(str).str.strip()
    req_df["day"] = req_df["day"].astype(str).str.strip()
    req_df["shift"] = req_df["shift"].astype(str).str.strip()
    pref_df["worker"] = pref_df["worker"].astype(str).str.strip()
    pref_df["day"] = pref_df["day"].astype(str).str.strip()
    pref_df["shift"] = pref_df["shift"].astype(str).str.strip()

    workers = workers_df["worker"].dropna().astype(str).tolist()
    if not workers:
        raise ValueError("לא נמצאו עובדים בטאב 'workers'.")

    # דרישות משמרות -> סלוטים
    req_df["required"] = pd.to_numeric(req_df["required"], errors="coerce").fillna(0).astype(int)

    shift_slots = []      # (day, shift, i)
    day_shift_pairs = []  # (day, shift) ייחודי

    for _, row in req_df.iterrows():
        day = str(row["day"])
        shift = str(row["shift"])
        req = int(row["required"])
        if req <= 0:
            continue

        pair = (day, shift)
        if pair not in day_shift_pairs:
            day_shift_pairs.append(pair)

        for i in range(req):
            shift_slots.append((day, shift, i))

    if not shift_slots:
        raise ValueError("לא נמצאו דרישות משמרות בטאב 'requirements' (required חייב להיות > 0).")

    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    # העדפות למילון
    pref_dict = {}
    for _, row in pref_df.iterrows():
        w = str(row["worker"])
        d = str(row["day"])
        s = str(row["shift"])
        try:
            p = int(row["preference"])
        except Exception:
            continue
        pref_dict[(w, d, s)] = p

    # worker_copies – רק צירופים שהעדפה שלהם >= 0
    worker_copies = []
    for w in workers:
        for (d, s) in day_shift_pairs:
            p = pref_dict.get((w, d, s), -1)
            if p >= 0:
                worker_copies.append((w, d, s))

    if not worker_copies:
        raise ValueError("לא נמצאו העדיפויות החוקיות (>=0) בטאב 'preferences'.")

    # מטריצת עלויות
    cost_matrix = []
    for w, d, s in worker_copies:
        row_costs = []
        for sd, ss, _ in shift_slots:
            if (d, s) == (sd, ss):
                pref = pref_dict.get((w, d, s), 0)
                # pref=0 אפשרי אבל "יקר", pref גבוה => עלות נמוכה
                row_costs.append(100 if pref == 0 else 4 - pref)
            else:
                row_costs.append(1e6)
        cost_matrix.append(row_costs)

    cost_matrix = np.array(cost_matrix, dtype=float)

    # הקצאה חמדנית
    row_ind, col_ind = simple_assignment(cost_matrix)

    assignments = []
    used_slots = set()
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in ordered_days} for w in workers}
    worker_day_shift_assigned = set()

    max_shifts_per_worker = len(shift_slots) // len(workers) + 1

    # סידור לפי עלות
    pairs = list(zip(row_ind, col_ind))
    pairs.sort(key=lambda x: cost_matrix[x[0], x[1]])

    # סיבוב ראשון – הקצאה "הוגנת" יחסית
    for r, c in pairs:
        worker, _, _ = worker_copies[r]
        slot_day, slot_shift, slot_i = shift_slots[c]
        slot = (slot_day, slot_shift, slot_i)

        if cost_matrix[r][c] >= 1e6:
            continue

        wds_key = (worker, slot_day, slot_shift)
        if wds_key in worker_day_shift_assigned:
            continue
        if slot in used_slots:
            continue
        if worker_shift_count[worker] >= max_shifts_per_worker:
            continue

        # בדיקת משמרות צמודות באותו יום
        try:
            current_shift_index = full_shifts.index(slot_shift)
        except ValueError:
            current_shift_index = 0

        if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[worker][slot_day]):
            continue

        used_slots.add(slot)
        worker_day_shift_assigned.add(wds_key)

        assignments.append({"שבוע": week_number, "יום": slot_day, "משמרת": slot_shift, "עובד": worker})
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][slot_day].append(slot_shift)

    # סיבוב שני – מילוי חורים (פחות קשוחים)
    remaining_slots = [slot for slot in shift_slots if slot not in used_slots]
    unassigned_pairs = set()

    for slot_day, slot_shift, slot_i in remaining_slots:
        assigned = False
        for w in workers:
            pref = pref_dict.get((w, slot_day, slot_shift), -1)
            if pref < 0:
                continue

            try:
                current_shift_index = full_shifts.index(slot_shift)
            except ValueError:
                current_shift_index = 0

            if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[w][slot_day]):
                continue

            wds_key = (w, slot_day, slot_shift)
            if wds_key in worker_day_shift_assigned:
                continue

            used_slots.add((slot_day, slot_shift, slot_i))
            worker_day_shift_assigned.add(wds_key)

            assignments.append({"שבוע": week_number, "יום": slot_day, "משמרת": slot_shift, "עובד": w})
            worker_shift_count[w] += 1
            worker_daily_shifts[w][slot_day].append(slot_shift)
            assigned = True
            break

        if not assigned:
            unassigned_pairs.add((slot_day, slot_shift))

    df = pd.DataFrame(assignments)
    if df.empty:
        raise ValueError("לא נוצר אף שיבוץ. בדוק את הנתונים בטאבים.")

    # סידור
    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x))
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]

    return df, unassigned_pairs


# =========================
# UI
# =========================
st.title("🛠️ מערכת שיבוץ משמרות (Excel)")

uploaded_file = st.file_uploader("העלה קובץ אקסל (xlsx) עם טאבים: workers / requirements / preferences", type=["xlsx"])
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if uploaded_file is None:
    st.info("העלה קובץ אקסל כדי להתחיל.")
    st.stop()

if st.button("🚀 בצע שיבוץ והוסף גיליון חדש לקובץ"):
    try:
        xls = pd.ExcelFile(uploaded_file)

        needed = {"workers", "requirements", "preferences"}
        existing = set(xls.sheet_names)
        missing = needed - existing
        if missing:
            raise ValueError(f"חסרים טאבים בקובץ: {sorted(list(missing))}. חייב: workers, requirements, preferences.")

        workers_df = pd.read_excel(xls, sheet_name="workers")
        req_df = pd.read_excel(xls, sheet_name="requirements")
        pref_df = pd.read_excel(xls, sheet_name="preferences")

        schedule_df, unassigned_pairs = build_schedule(workers_df, req_df, pref_df, int(week_number))

        # לתצוגה יפה
        schedule_df = schedule_df.reset_index(drop=True)
        schedule_df.index += 1

        st.success("✅ השיבוץ הוכן בהצלחה!")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned_pairs:
            for d, s in sorted(list(unassigned_pairs)):
                st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

        # כתיבה לקובץ חדש עם טאבים מקוריים + טאב שבוע חדש
        new_sheet_name = f"שבוע {int(week_number)}"
        original_sheet_names = xls.sheet_names

        if new_sheet_name in original_sheet_names:
            new_sheet_name = f"{new_sheet_name} (2)"

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet in original_sheet_names:
                df_old = pd.read_excel(xls, sheet_name=sheet)
                df_old.to_excel(writer, sheet_name=sheet, index=False)

            schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

        output.seek(0)

        st.download_button(
            label="⬇️ הורד את הקובץ המעודכן (עם היסטוריית השבועות)",
            data=output,
            file_name=uploaded_file.name.replace(".xlsx", "") + "_with_schedule.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)
