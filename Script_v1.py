import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import os
from datetime import datetime

st.set_page_config(page_title="Bank Reconciliation Tool", layout="wide")
st.title("🔄 Bank Statement Reconciliation Tool")

def extract_utr(text):
    """Extract the first 12-digit number from a string, even if surrounded by slashes, dashes, or spaces."""
    if pd.isna(text):
        return None
    match = re.search(r"(?<!\d)(\d{12})(?!\d)", str(text))
    return match.group(1) if match else None

def make_unique_columns(df):
    cols = pd.Series(df.columns)
    counts = {}
    new_cols = []
    for col in cols:
        if col not in counts:
            counts[col] = 1
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}.{counts[col]}")
    df.columns = new_cols
    return df

def load_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".csv":
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    elif ext in [".xls", ".xlsx"]:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    else:
        raise Exception("Unsupported file type")
    df = make_unique_columns(df)
    return df

def clean_amount(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("₹", "").strip()
    if s in ["", "-", "--"]:
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan

def reco_duplicate_flag(df, utr_col):
    utr_counts = df[utr_col].value_counts()
    df['Duplicate Remark'] = df[utr_col].map(lambda x: "Duplicate Transaction" if utr_counts.get(x, 0) > 1 and pd.notna(x) else "")
    return df

def get_effective_amount(row, debit_col, credit_col, amount_col):
    credit = clean_amount(row[credit_col]) if credit_col and credit_col in row and row[credit_col] is not None else None
    debit = clean_amount(row[debit_col]) if debit_col and debit_col in row and row[debit_col] is not None else None
    amount = clean_amount(row[amount_col]) if amount_col and amount_col in row and row[amount_col] is not None else None

    if credit is not None and credit != 0:
        return credit
    if debit is not None and debit != 0:
        return debit
    if amount is not None and amount != 0:
        return amount
    if (credit == 0 or credit is None) and (debit == 0 or debit is None) and (amount == 0 or amount is None):
        return 0.0
    return np.nan

def get_all_possible_amounts(row, debit_col, credit_col, amount_col):
    """Return a set of all possible (non-NaN) amounts from debit, credit, and amount columns for matching."""
    vals = set()
    if debit_col and debit_col in row:
        val = clean_amount(row[debit_col])
        if not pd.isna(val):
            vals.add(val)
    if credit_col and credit_col in row:
        val = clean_amount(row[credit_col])
        if not pd.isna(val):
            vals.add(val)
    if amount_col and amount_col in row:
        val = clean_amount(row[amount_col])
        if not pd.isna(val):
            vals.add(val)
    return vals

# ... [rest of your code remains unchanged above] ...

def match_transactions(
    reco_df, bank_dfs,
    reco_utr, reco_amt, reco_date, reco_desc,
    bank_utr_list, bank_amt_list, bank_date_list, bank_desc_list,
    reco_file, bank_file_list,
    optional_mapped,
    reco_debit_col, reco_credit_col, reco_amount_col,
    bank_debit_cols, bank_credit_cols, bank_amount_cols
):
    extra_display_names = list(optional_mapped.keys())
    extra_col_names = [optional_mapped[disp] for disp in extra_display_names]
    # Add Debit, Credit, Amount columns (raw, as per mapping) after "Amount" in output
    extra_bank_export = ["Debit", "Credit", "Amount"]
    extra_reco_export = ["Debit", "Credit", "Amount"]
    output_cols = ["Date", "Description", "Amount", "Debit", "Credit", "Raw Amount", "Source", "UTR", "Remark"] + extra_display_names

    # Instead of using a dict, collect all bank rows into a list (to allow duplicate UTRs)
    all_bank_rows = []
    for df, bank_utr, bank_amt, bank_date, bank_desc, bank_file, bank_debit_col, bank_credit_col, bank_amount_col in zip(
        bank_dfs, bank_utr_list, bank_amt_list, bank_date_list, bank_desc_list, bank_file_list, bank_debit_cols, bank_credit_cols, bank_amount_cols
    ):
        for idx, row in df.iterrows():
            utr = row[bank_utr]
            # For matching, store all possible amounts
            all_amounts = get_all_possible_amounts(row, bank_debit_col, bank_credit_col, bank_amount_col)
            amt = get_effective_amount(row, bank_debit_col, bank_credit_col, bank_amount_col)
            entry = {
                "Date": row[bank_date],
                "Description": row[bank_desc],
                "Amount": amt,
                "Debit": row[bank_debit_col] if bank_debit_col else "",
                "Credit": row[bank_credit_col] if bank_credit_col else "",
                "Raw Amount": row[bank_amount_col] if bank_amount_col else "",
                "Source": bank_file,
                "UTR": utr,
                "__all_amounts__": all_amounts  # for internal matching use only
            }
            all_bank_rows.append(entry)

    from collections import defaultdict
    bank_utr_to_rows = defaultdict(list)
    for row in all_bank_rows:
        bank_utr_to_rows[row["UTR"]].append(row)

    reco_rows = []
    for idx, row in reco_df.iterrows():
        utr = row[reco_utr]
        base_remark = row.get("Duplicate Remark", "")
        if base_remark:
            remark = base_remark
        elif pd.isna(utr) or utr == "":
            remark = "No UTR"
        else:
            bank_rows = bank_utr_to_rows.get(utr, [])
            if not bank_rows:
                remark = "UTR not in Bank Statement"
            else:
                reco_amounts = get_all_possible_amounts(row, reco_debit_col, reco_credit_col, reco_amount_col)
                matched = False
                for b_row in bank_rows:
                    bank_amounts = b_row["__all_amounts__"]
                    if any(r_amt == b_amt for r_amt in reco_amounts for b_amt in bank_amounts):
                        matched = True
                        break
                if matched:
                    remark = "Matched"
                else:
                    remark = "Amount Different"
        entry = {
            "Date": row[reco_date],
            "Description": row[reco_desc],
            "Amount": row[reco_amt],
            "Debit": row[reco_debit_col] if reco_debit_col else "",
            "Credit": row[reco_credit_col] if reco_credit_col else "",
            "Raw Amount": row[reco_amount_col] if reco_amount_col else "",
            "Source": reco_file,
            "UTR": utr,
            "Remark": remark
        }
        for disp, colname in zip(extra_display_names, extra_col_names):
            entry[disp] = row.get(colname, "")
        reco_rows.append(entry)

    # Export columns: show both mapped and extra columns
    export_bank_cols = ["Date", "Description", "Amount", "Debit", "Credit", "Raw Amount", "Source", "UTR", "Remark"] + extra_display_names
    export_reco_cols = ["Date", "Description", "Amount", "Debit", "Credit", "Raw Amount", "Source", "UTR", "Remark"] + extra_display_names

    reco_report = pd.DataFrame(reco_rows)
    for col in export_reco_cols:
        if col not in reco_report.columns:
            reco_report[col] = ""
    reco_report = reco_report[export_reco_cols]

    bank_output_rows = []
    reco_utr_to_amounts = defaultdict(set)
    for idx, row in reco_df.iterrows():
        reco_utr_val = row[reco_utr]
        amounts = get_all_possible_amounts(row, reco_debit_col, reco_credit_col, reco_amount_col)
        reco_utr_to_amounts[reco_utr_val].update(amounts)
    for row in all_bank_rows:
        utr = row["UTR"]
        bank_amounts = row["__all_amounts__"]
        if pd.isna(utr) or utr == "":
            remark = "No UTR"
        elif utr not in reco_utr_to_amounts:
            remark = "UTR not in Reco Sheet"
        elif any(amt in reco_utr_to_amounts[utr] for amt in bank_amounts):
            remark = "Matched"
        else:
            remark = "Amount Different"
        entry = row.copy()
        entry.pop("__all_amounts__", None)
        entry["Remark"] = remark
        bank_output_rows.append(entry)

    bank_report = pd.DataFrame(bank_output_rows)
    for col in export_bank_cols:
        if col not in bank_report.columns:
            bank_report[col] = ""
    bank_report = bank_report[export_bank_cols]

    return bank_report, reco_report

def to_excel(bank_df, reco_df):
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    with pd.ExcelWriter(output.name, engine='openpyxl') as writer:
        bank_df.to_excel(writer, sheet_name="Bank Statement Entries", index=False)
        reco_df.to_excel(writer, sheet_name="Reco Sheet Entries", index=False)
    return output.name

# ... [rest of your code remains unchanged below] ...

# RESET BUTTON
if st.button("Reset All"):
    st.session_state.clear()
    st.experimental_rerun()

if "reco_file" not in st.session_state:
    st.session_state["reco_file"] = None
if "bank_files" not in st.session_state:
    st.session_state["bank_files"] = None

with st.form("upload_form"):
    st.header("Step 1: Upload Files")
    reco_file = st.file_uploader("Upload Reco Sheet (.xls, .xlsx, .csv)", type=['xls','xlsx','csv'], key="recofile")
    bank_files = st.file_uploader("Upload Bank Statement(s) (.xls, .xlsx, .csv)", accept_multiple_files=True, type=['xls','xlsx','csv'], key="bankfiles")
    submitted = st.form_submit_button("Next (Map Columns)")
    if submitted:
        if reco_file and bank_files:
            st.session_state["reco_file"] = reco_file
            st.session_state["bank_files"] = bank_files
        else:
            st.warning("Please upload both the Reco Sheet and at least one Bank Statement.")

if st.session_state.get("reco_file") and st.session_state.get("bank_files"):
    reco_file = st.session_state["reco_file"]
    bank_files = st.session_state["bank_files"]
    reco_df = load_file(reco_file)
    st.markdown("---")
    st.subheader("Step 2: Map Columns for Reco Sheet")
    columns = list(reco_df.columns)
    st.write("Columns found in reco file:", columns)
    reco_text_col = st.selectbox("Column containing UTR (Remark/Description/Particulars)", columns, key="reco_text_col2")
    reco_debit_col = st.selectbox("Debit column (leave blank if not applicable)", [""] + columns, key="reco_debit_col2")
    reco_credit_col = st.selectbox("Credit column (leave blank if not applicable)", [""] + columns, key="reco_credit_col2")
    reco_amount_col = st.selectbox("Amount column (leave blank if not applicable)", [""] + columns, key="reco_amount_col2")
    reco_date_col = st.selectbox("Date column", columns, key="reco_date_col2")

    # Optional columns mapping UI (visible after reco columns are loaded)
    st.markdown("**Map extra columns to include in the Reco Sheet Entries (optional):**")
    optional_fields = ["Username", "Trans. ID", "Transaction Type", "Approved By"]
    optional_mapped = {}
    for field in optional_fields:
        sel = st.selectbox(
            f"Column for {field} (optional)", [""] + columns, key=f"reco_{field}_col"
        )
        if sel and sel.strip():
            optional_mapped[field] = sel

    preview_cols = [reco_text_col]
    for col in [reco_debit_col, reco_credit_col, reco_amount_col]:
        if col and col.strip():
            preview_cols.append(col)
    preview_cols.append(reco_date_col)
    # Show sample of mapped optional fields, if mapped
    for field, sel in optional_mapped.items():
        if sel and sel.strip():
            preview_cols.append(sel)
    preview_cols = [x for i, x in enumerate(preview_cols) if x not in preview_cols[:i] and x != ""]
    reco_preview = reco_df[preview_cols].head(5)
    st.write("Sample data from Reco Sheet:")
    st.dataframe(reco_preview)
    st.markdown("---")
    st.subheader("Step 3: Map Columns for Each Bank Statement")

    bank_df_list = []
    bank_utr_list = []
    bank_amt_list = []
    bank_date_list = []
    bank_desc_list = []
    bank_file_list = []
    bank_debit_cols = []
    bank_credit_cols = []
    bank_amount_cols = []
    for i, f in enumerate(bank_files):
        st.markdown(f"##### Bank Statement: `{f.name}`")
        df = load_file(f)
        columns = list(df.columns)
        st.write("Columns found in this bank file:", columns)
        bank_text_col = st.selectbox(f"Column containing UTR (Remark/Description/Particulars) [{f.name}]", columns, key=f"bank_text_col_{i}_new")
        bank_debit_col = st.selectbox(f"Debit column (leave blank if not applicable) [{f.name}]", [""] + columns, key=f"bank_debit_col_{i}_new")
        bank_credit_col = st.selectbox(f"Credit column (leave blank if not applicable) [{f.name}]", [""] + columns, key=f"bank_credit_col_{i}_new")
        bank_amount_col = st.selectbox(f"Amount column (leave blank if not applicable) [{f.name}]", [""] + columns, key=f"bank_amount_col_{i}_new")
        bank_date_col = st.selectbox(f"Date column [{f.name}]", columns, key=f"bank_date_col_{i}_new")
        preview_cols = [bank_text_col]
        for col in [bank_debit_col, bank_credit_col, bank_amount_col]:
            if col and col.strip():
                preview_cols.append(col)
        preview_cols.append(bank_date_col)
        preview_cols = [x for i, x in enumerate(preview_cols) if x not in preview_cols[:i] and x != ""]
        bank_preview = df[preview_cols].head(5)
        st.write("Sample data:", bank_preview)
        # Apply UTR and Amount logic to bank df for later matching
        df['Bank UTR'] = df[bank_text_col].astype(str).apply(extract_utr)
        df['Bank Amount'] = df.apply(
            lambda row: get_effective_amount(
                row,
                bank_debit_col if bank_debit_col else None,
                bank_credit_col if bank_credit_col else None,
                bank_amount_col if bank_amount_col else None
            ), axis=1
        )
        df['Bank Date'] = df[bank_date_col]
        df['Bank Description'] = df[bank_text_col]
        bank_df_list.append(df)
        bank_utr_list.append('Bank UTR')
        bank_amt_list.append('Bank Amount')
        bank_date_list.append('Bank Date')
        bank_desc_list.append('Bank Description')
        bank_file_list.append(f.name)
        bank_debit_cols.append(bank_debit_col if bank_debit_col else None)
        bank_credit_cols.append(bank_credit_col if bank_credit_col else None)
        bank_amount_cols.append(bank_amount_col if bank_amount_col else None)
    st.markdown("---")
    process_btn = st.button("Run Reconciliation and Export Report")
    if process_btn:
        reco_df['Reco UTR'] = reco_df[reco_text_col].astype(str).apply(extract_utr)
        reco_df['Reco Amount'] = reco_df.apply(
            lambda row: get_effective_amount(
                row,
                reco_debit_col if reco_debit_col else None,
                reco_credit_col if reco_credit_col else None,
                reco_amount_col if reco_amount_col else None
            ), axis=1
        )
        reco_df['Reco Date'] = reco_df[reco_date_col]
        reco_df['Reco Description'] = reco_df[reco_text_col]
        reco_df = reco_duplicate_flag(reco_df, 'Reco UTR')

        bank_report, reco_report = match_transactions(
            reco_df, bank_df_list,
            'Reco UTR', 'Reco Amount', 'Reco Date', 'Reco Description',
            bank_utr_list, bank_amt_list, bank_date_list, bank_desc_list,
            reco_file.name, bank_file_list,
            optional_mapped,
            reco_debit_col if reco_debit_col else None,
            reco_credit_col if reco_credit_col else None,
            reco_amount_col if reco_amount_col else None,
            bank_debit_cols,
            bank_credit_cols,
            bank_amount_cols
        )
        out_file = to_excel(bank_report, reco_report)
        st.success("Reconciliation complete! Download your report below.")
        with open(out_file, "rb") as f:
            st.download_button(
                label="Download Excel Report",
                data=f,
                file_name=f"Reconciliation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.write("Preview of Bank Statement Entries:")
        st.dataframe(bank_report.head(10))
        st.write("Preview of Reco Sheet Entries:")
        st.dataframe(reco_report.head(10))