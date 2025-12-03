"""
Pension Calculation System - Streamlit App (Deployment Ready)
--------------------------------------------------------------
This module calculates pension benefits for retirees based on Nigerian pension regulations.
It computes recommended lump sum payments and monthly pension amounts based on:
- RSA (Retirement Savings Account) balance
- Actuarial tables (gender and frequency dependent)
- Statutory charges and interest rates
- Minimum pension payout requirements (50% of final monthly salary)
"""

import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Statutory charges (as per pension regulations)
MANAGEMENT_CHARGES = 0.065
REGULATORY_CHARGES = 0.01
INTEREST_RATE = 0.105
DISCOUNT_RATE = 0.08

# Pension calculation parameters
MIN_PENSION_PAYOUT_PERCENT = 0.5  # 50% of final monthly salary
FINAL_SALARY_PERCENT = 1
MAX_MONTHLY_INCREASE = 0.5  # 50% maximum increase

# Derived values
INTEREST_RATE_NET_CHARGES = INTEREST_RATE * (1 - MANAGEMENT_CHARGES - REGULATORY_CHARGES)

# Actuarial adjustment factor (11/24 represents mid-period adjustment)
ACTUARIAL_MID_PERIOD_ADJUSTMENT = 11 / 24

# Frequency multiplier for annuity period calculation
FREQUENCY_MULTIPLIER = 2


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Pension Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1e3a8a;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #2563eb;
        margin-top: 2rem;
    }
    h3 {
        color: #3b82f6;
    }
    .result-box {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_pension_data():
    """
    Load all required CSV files for pension calculations.
    Uses relative paths for deployment compatibility.
    Checks multiple locations: data/ folder, root folder, and current directory.
    
    Returns:
        tuple: (male12, female12, male4, female4, salarystructure) DataFrames or None if files not found
    """
    # Try multiple locations
    search_locations = [
        "data",           # data/ subfolder
        ".",              # root/current directory
        ""                # same directory as script
    ]
    
    file_names = {
        'male12': "Male12.csv",
        'female12': "Female12.csv",
        'male4': "Male4.csv",
        'female4': "Female4.csv",
        'salarystructure': "SalaryStructure.csv"
    }
    
    # Try to find files in each location
    file_paths = {}
    for location in search_locations:
        temp_paths = {
            key: os.path.join(location, fname) if location else fname
            for key, fname in file_names.items()
        }
        
        # Check if all files exist in this location
        if all(os.path.exists(path) for path in temp_paths.values()):
            file_paths = temp_paths
            break
    
    # If no location has all files, report missing
    if not file_paths:
        # Check which files are missing from all locations
        missing_files = list(file_names.keys())
        return None, missing_files
    
    # Try to load the files
    try:
        male12 = pd.read_csv(file_paths['male12'])
        female12 = pd.read_csv(file_paths['female12'])
        male4 = pd.read_csv(file_paths['male4'])
        female4 = pd.read_csv(file_paths['female4'])
        salarystructure = pd.read_csv(file_paths['salarystructure'])
        
        # Ensure Annual Salary is float type
        salarystructure['Annual Salary'] = salarystructure['Annual Salary'].astype(float)
        
        return (male12, female12, male4, female4, salarystructure), None
    
    except Exception as e:
        st.error(f"❌ Error loading data files: {e}")
        return None, [str(e)]


def load_uploaded_files(male12_file, female12_file, male4_file, female4_file, salary_file):
    """Load data from uploaded files."""
    try:
        male12 = pd.read_csv(male12_file)
        female12 = pd.read_csv(female12_file)
        male4 = pd.read_csv(male4_file)
        female4 = pd.read_csv(female4_file)
        salarystructure = pd.read_csv(salary_file)
        
        # Ensure Annual Salary is float type
        salarystructure['Annual Salary'] = salarystructure['Annual Salary'].astype(float)
        
        return male12, female12, male4, female4, salarystructure
    except Exception as e:
        st.error(f"❌ Error processing uploaded files: {e}")
        return None


# ============================================================================
# DATE & TIME UTILITIES
# ============================================================================

def datedif(start_date, end_date, unit):
    """
    Calculate the difference between two dates in various units.
    Mimics Excel's DATEDIF function.
    """
    delta = relativedelta(end_date, start_date)

    if unit == "Y":
        return delta.years
    elif unit == "M":
        return delta.years * 12 + delta.months
    elif unit == "D":
        return (end_date - start_date).days
    elif unit == "YM":
        return delta.months
    elif unit == "MD":
        return delta.days
    elif unit == "YD":
        delta_this_year = datetime(end_date.year, end_date.month, end_date.day) - datetime(end_date.year, start_date.month, start_date.day)
        return delta_this_year.days if delta_this_year.days >= 0 else (datetime(end_date.year + 1, start_date.month, start_date.day) - datetime(end_date.year, end_date.month, end_date.day)).days
    else:
        raise ValueError("Invalid unit. Use 'Y', 'M', 'D', 'YM', 'MD', or 'YD'.")


# ============================================================================
# FINANCIAL CALCULATION FUNCTIONS
# ============================================================================

def pmt(rate, nper, pv, fv=0, when=0):
    """Calculate payment amount for an annuity."""
    if rate == 0:
        return -(pv + fv) / nper
    else:
        factor = (1 + rate) ** nper
        return -(pv * factor + fv) / ((1 + rate * when) * (factor - 1) / rate)


def pv(rate, nper, pmt, fv=0, when=0):
    """Calculate present value of an annuity."""
    if rate == 0:
        return -pmt * nper - fv
    else:
        factor = (1 + rate) ** nper
        return -(pmt * (1 + rate * when) * (factor - 1) / rate + fv) / factor


# ============================================================================
# ACTUARIAL TABLE LOOKUP
# ============================================================================

def lookup_annuity_factor(gender, frequency, current_age, male4, male12, female4, female12):
    """Look up the annuity factor (ax) from actuarial tables."""
    # Select the appropriate actuarial table
    if gender == "Male" and frequency == 4:
        table = male4
    elif gender == "Male" and frequency == 12:
        table = male12
    elif gender == "Female" and frequency == 4:
        table = female4
    elif gender == "Female" and frequency == 12:
        table = female12
    else:
        raise ValueError("Invalid combination of gender and frequency.")
    
    # Perform the lookup
    match = table.loc[table['age'] == current_age]

    if not match.empty:
        annuity_factor = match.iloc[0]['ax']
        return annuity_factor
    else:
        raise ValueError(f"Age {current_age} not found in selected table.")


# ============================================================================
# PENSION CALCULATION LOGIC
# ============================================================================

def determine_recommended_pension(current_pension, minimum_pension_payout, max_pension_possible):
    """Determine the recommended pension amount."""
    if current_pension <= minimum_pension_payout:
        return min(max_pension_possible, minimum_pension_payout)
    else:
        return min(max_pension_possible, current_pension)


def calculate_additional_pension_scenario(new_pension, other_pension, current_pension, 
                                         max_pension_possible, other_max_pension_possible,
                                         recommended_pension, other_recommended_pension):
    """Determine which pension scenario applies."""
    if new_pension < current_pension:
        return [
            other_max_pension_possible,
            other_recommended_pension,
            other_pension + current_pension
        ]
    else:
        return [
            max_pension_possible,
            recommended_pension,
            new_pension
        ]


def get_annual_salary(salary_structure, grade_level, step, df):
    """Look up annual salary from salary structure table."""
    result = df[
        (df["Salary Structure"].str.lower() == salary_structure.lower()) &
        (df["Grade Level"] == str(grade_level)) &
        (df["Step"] == str(step))
    ]
    
    if not result.empty:
        return result.iloc[0]["Annual Salary"]
    else:
        return None


# ============================================================================
# FILE UPLOAD INTERFACE
# ============================================================================

def show_file_upload_interface():
    """Display file upload interface when data files are not found."""
    st.warning("⚠️ Required data files not found in the repository.")
    st.info("Please upload the required CSV files to proceed with calculations.")
    
    with st.expander("📁 Upload Required Data Files", expanded=True):
        st.markdown("""
        **Required Files:**
        1. Male12.csv - Male actuarial table (12 payments/year)
        2. Female12.csv - Female actuarial table (12 payments/year)
        3. Male4.csv - Male actuarial table (4 payments/year)
        4. Female4.csv - Female actuarial table (4 payments/year)
        5. SalaryStructure.csv - Public sector salary structure
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            male12_file = st.file_uploader("Male12.csv", type=['csv'], key="male12")
            female12_file = st.file_uploader("Female12.csv", type=['csv'], key="female12")
            male4_file = st.file_uploader("Male4.csv", type=['csv'], key="male4")
        
        with col2:
            female4_file = st.file_uploader("Female4.csv", type=['csv'], key="female4")
            salary_file = st.file_uploader("SalaryStructure.csv", type=['csv'], key="salary")
        
        if all([male12_file, female12_file, male4_file, female4_file, salary_file]):
            if st.button("✅ Load Files and Continue"):
                result = load_uploaded_files(male12_file, female12_file, male4_file, female4_file, salary_file)
                if result:
                    st.session_state.data_loaded = True
                    st.session_state.pension_data = result
                    st.success("✅ All files loaded successfully!")
                    st.rerun()
        else:
            st.warning("Please upload all 5 required CSV files.")
            return None
    
    return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("💰 Pension Calculation System")
    st.markdown("Calculate pension benefits based on Nigerian pension regulations")
    
    # Load data or show upload interface
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading pension data..."):
            result, missing_files = load_pension_data()
        
        if result is None:
            st.error(f"❌ Missing files: {', '.join(missing_files) if missing_files else 'Unknown error'}")
            pension_data = show_file_upload_interface()
            if pension_data is None:
                st.stop()
        else:
            male12, female12, male4, female4, salarystructure = result
            st.session_state.data_loaded = True
            st.session_state.pension_data = (male12, female12, male4, female4, salarystructure)
    else:
        male12, female12, male4, female4, salarystructure = st.session_state.pension_data
    
    # Initialize session state for multi-step workflow
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'calculation_done' not in st.session_state:
        st.session_state.calculation_done = False
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.info("Fill in all required information to calculate pension benefits")
    
    # Progress indicator
    progress_text = f"Step {st.session_state.step} of 3"
    st.sidebar.write(progress_text)
    st.sidebar.progress(st.session_state.step / 3)
    
    # Display app info in sidebar
    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        **Pension Calculator v1.0**
        
        This application calculates:
        - Maximum possible pension
        - Recommended lump sum payment
        - New monthly pension amount
        
        Based on Nigerian pension regulations and actuarial tables.
        """)
    
    # ========================================================================
    # STEP 1: Basic Client Information
    # ========================================================================
    
    if st.session_state.step >= 1:
        st.header("Step 1: Basic Client Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                key="gender"
            )
            
            dob = st.date_input(
                "Date of Birth",
                min_value=datetime(1940, 1, 1).date(),
                max_value=datetime.today().date(),
                key="dob"
            )
            
            retirement_date = st.date_input(
                "Retirement Date",
                min_value=datetime(2000, 1, 1).date(),
                max_value=datetime.today().date(),
                key="retirement_date"
            )
        
        with col2:
            sector = st.selectbox(
                "Sector",
                options=["Public", "Private"],
                key="sector"
            )
            
            request_date = st.date_input(
                "Request Date",
                min_value=datetime(2000, 1, 1).date(),
                max_value=datetime.today().date(),
                value=datetime.today().date(),
                key="request_date"
            )
            
            frequency = st.selectbox(
                "Payment Frequency",
                options=[12, 4],
                format_func=lambda x: "Monthly (12)" if x == 12 else "Quarterly (4)",
                key="frequency"
            )
        
        if st.button("Next →", key="step1_next"):
            # Validate dates
            if retirement_date <= dob:
                st.error("❌ Retirement date must be after date of birth")
            elif request_date < retirement_date:
                st.error("❌ Request date must be on or after retirement date")
            else:
                st.session_state.step = 2
                st.rerun()
    
    # ========================================================================
    # STEP 2: Financial Information
    # ========================================================================
    
    if st.session_state.step >= 2:
        st.header("Step 2: Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rsa_balance_before_inflow = st.number_input(
                "RSA Balance Before Inflow (₦)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                key="rsa_before"
            )
            
            additional_inflow = st.number_input(
                "Additional Inflow (₦)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                key="additional_inflow"
            )
            
            rsa_balance = st.number_input(
                "Current RSA Balance (₦)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f",
                key="rsa_balance"
            )
        
        with col2:
            current_pension = st.number_input(
                "Current Pension (₦)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f",
                key="current_pension"
            )
            
            # Salary input based on sector
            if st.session_state.sector == "Public":
                st.subheader("Public Sector Salary Lookup")
                
                salary_structure = st.text_input(
                    "Salary Structure (e.g., CONPOSS)",
                    key="salary_structure"
                )
                
                col_grade, col_step = st.columns(2)
                with col_grade:
                    grade_level = st.text_input("Grade Level", key="grade_level")
                with col_step:
                    step_input = st.text_input("Step", key="step_input")
                
                if salary_structure and grade_level and step_input:
                    validated_salary = get_annual_salary(
                        salary_structure, grade_level, step_input, salarystructure
                    )
                    if validated_salary:
                        st.success(f"✅ Annual Salary Found: ₦{validated_salary:,.2f}")
                    else:
                        st.error("❌ No matching salary record found")
                        validated_salary = None
                else:
                    validated_salary = None
            else:
                validated_salary = st.number_input(
                    "Annual Salary (₦)",
                    min_value=0.0,
                    value=0.0,
                    step=10000.0,
                    format="%.2f",
                    key="annual_salary"
                )
        
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", key="step2_back"):
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("Calculate →", key="step2_calculate"):
                if validated_salary is None or validated_salary == 0:
                    st.error("❌ Please provide a valid annual salary")
                elif rsa_balance == 0:
                    st.error("❌ RSA balance cannot be zero")
                else:
                    st.session_state.validated_salary = validated_salary
                    st.session_state.step = 3
                    st.rerun()
    
    # ========================================================================
    # STEP 3: Calculation & Results
    # ========================================================================
    
    if st.session_state.step >= 3:
        st.header("Step 3: Pension Calculation Results")
        
        # Retrieve all inputs from session state
        gender = st.session_state.gender
        sector = st.session_state.sector
        dob = datetime.combine(st.session_state.dob, datetime.min.time())
        retirement_date = datetime.combine(st.session_state.retirement_date, datetime.min.time())
        request_date = datetime.combine(st.session_state.request_date, datetime.min.time())
        rsa_balance_before_inflow = st.session_state.rsa_before
        additional_inflow = st.session_state.additional_inflow
        rsa_balance = st.session_state.rsa_balance
        current_pension = st.session_state.current_pension
        frequency = st.session_state.frequency
        validated_salary = st.session_state.validated_salary
        
        # Calculate age and actuarial parameters
        current_age = datedif(dob, request_date, "Y")
        
        try:
            annuity_factor = lookup_annuity_factor(
                gender, frequency, current_age, male4, male12, female4, female12
            )
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            if st.button("← Back to Financial Info", key="error_back"):
                st.session_state.step = 2
                st.rerun()
            st.stop()
        
        # Calculate annuity period parameters
        adjusted_annuity = annuity_factor - ACTUARIAL_MID_PERIOD_ADJUSTMENT
        total_payment_periods = FREQUENCY_MULTIPLIER * frequency * adjusted_annuity
        monthly_interest_rate = INTEREST_RATE_NET_CHARGES / 12
        years_in_retirement = datedif(retirement_date, request_date, "Y")
        
        # Calculate minimum pension payout
        minimum_pension_payout = (validated_salary / frequency) * MAX_MONTHLY_INCREASE
        
        # Calculate maximum possible pensions
        max_pension_possible = -pmt(monthly_interest_rate, total_payment_periods, 
                                    rsa_balance, fv=0, when=1)
        other_max_pension_possible = -pmt(monthly_interest_rate, total_payment_periods, 
                                          additional_inflow, fv=0, when=1)
        
        # Determine recommended pensions
        recommended_pension = determine_recommended_pension(
            current_pension, minimum_pension_payout, max_pension_possible
        )
        other_recommended_pension = determine_recommended_pension(
            current_pension, minimum_pension_payout, other_max_pension_possible
        )
        
        # Calculate recommended lumpsum
        recommended_lumpsum = max(0, (rsa_balance + pv(monthly_interest_rate, 
                                                       total_payment_periods, 
                                                       recommended_pension, 
                                                       fv=0, when=1)))
        
        # Display recommended lumpsum and get negotiated value
        st.subheader("Lumpsum Negotiation")
        st.info(f"💡 **Recommended Lumpsum:** ₦{recommended_lumpsum:,.2f}")
        
        negotiated_lumpsum = st.number_input(
            "Enter Negotiated Lumpsum (₦)",
            min_value=0.0,
            max_value=float(recommended_lumpsum),
            value=0.0,
            step=1000.0,
            format="%.2f",
            key="negotiated_lumpsum",
            help="Cannot exceed recommended lumpsum"
        )
        
        if negotiated_lumpsum > recommended_lumpsum:
            st.error("❌ Negotiated lumpsum cannot exceed recommended lumpsum")
        
        # Calculate final pensions
        new_pension = -pmt(monthly_interest_rate, total_payment_periods, 
                          (rsa_balance - negotiated_lumpsum), fv=0, when=1)
        other_pension = -pmt(monthly_interest_rate, total_payment_periods, 
                            (additional_inflow - negotiated_lumpsum), fv=0, when=1)
        
        # Determine final scenario
        output_values = calculate_additional_pension_scenario(
            new_pension, other_pension, current_pension,
            max_pension_possible, other_max_pension_possible,
            recommended_pension, other_recommended_pension
        )
        
        # Display Results
        st.markdown("---")
        st.subheader("📊 Final Calculation Results")
        
        # Create results in a nice layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Validated Final Salary", f"₦{validated_salary:,.2f}")
            st.metric("50% of Monthly Final Salary", f"₦{minimum_pension_payout:,.2f}")
            st.metric("Current Pension", f"₦{current_pension:,.2f}")
        
        with col2:
            st.metric("Max Pension Possible", f"₦{output_values[0]:,.2f}")
            st.metric("Recommended Pension", f"₦{output_values[1]:,.2f}")
            st.metric("Recommended Lumpsum", f"₦{recommended_lumpsum:,.2f}")
        
        with col3:
            st.metric("Negotiated Lumpsum", f"₦{negotiated_lumpsum:,.2f}")
            st.metric(
                "New Monthly Pension", 
                f"₦{output_values[2]:,.2f}",
                delta=f"₦{output_values[2] - current_pension:,.2f}"
            )
        
        # Summary box
        st.markdown("---")
        st.markdown("""
        <div class="result-box">
            <h3>✅ Calculation Summary</h3>
            <ul>
                <li><strong>Client Age:</strong> {} years</li>
                <li><strong>Years in Retirement:</strong> {} years</li>
                <li><strong>Payment Frequency:</strong> {}</li>
                <li><strong>Annuity Factor:</strong> {:.4f}</li>
                <li><strong>Total Payment Periods:</strong> {:.2f}</li>
            </ul>
        </div>
        """.format(
            current_age,
            years_in_retirement,
            "Monthly" if frequency == 12 else "Quarterly",
            annuity_factor,
            total_payment_periods
        ), unsafe_allow_html=True)
        
        # Action buttons
        col_back, col_new = st.columns([1, 1])
        with col_back:
            if st.button("← Back to Financial Info", key="step3_back"):
                st.session_state.step = 2
                st.rerun()
        with col_new:
            if st.button("🔄 New Calculation", key="new_calc"):
                # Clear all session state except data
                keys_to_keep = ['data_loaded', 'pension_data']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                st.rerun()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
