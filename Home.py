from pathlib import Path

import streamlit as st

# Configure the page with optimized settings
st.set_page_config(
    page_title="Investment Analysis Hub",
    layout="wide",
    initial_sidebar_state="expanded",  # Better UX by showing sidebar initially
)


def main() -> None:
    """Main function to render the home page of Investment Analysis Hub"""
    # Header Section
    render_header()

    # Display available pages section
    render_available_pages()

    # Footer
    st.markdown("---")
    st.success("👉 Use the **sidebar** to navigate to each section.")

    # Add footer
    render_footer()


def render_header() -> None:
    """Renders the application header and introduction text"""
    # Title with emoji for visual appeal
    st.title("📊 Investment Analysis Hub")
    st.subheader(
        "Unlock Insights. Make Smarter Investment Decisions.", divider="rainbow"
    )

    # Introduction text
    st.markdown("""
    ##### Welcome to the **Investment Analysis Hub** – your central place to explore investment data, monitor market trends, and gain financial insights.
    """)


def render_available_pages() -> None:
    """Scans for and displays available pages in the application"""
    st.markdown("##### 📄 Available Sections (visible in sidebar):")

    # Use Path for more reliable cross-platform path handling
    pages_dir = Path("pages")

    if pages_dir.exists() and pages_dir.is_dir():
        # Get all Python files that don't start with underscore
        page_files = sorted(
            [
                f
                for f in pages_dir.iterdir()
                if f.is_file() and f.suffix == ".py" and not f.name.startswith("_")
            ]
        )

        # No pages found
        if not page_files:
            st.info("No analysis modules have been added yet.")
            return

        # Display each page with an emoji bullet
        for page_path in page_files:
            # Extract name without extension
            name = page_path.stem

            # Format display name: remove leading numbers/underscores and replace underscores with spaces
            display_name = " ".join(
                part for part in name.split("_") if part and not part[0].isdigit()
            ).title()

            st.markdown(f"-📍**{display_name}**")
    else:
        st.warning(
            "Pages directory not found. Please create a `pages/` directory with your analysis modules."
        )
        st.info(
            "Learn more about Streamlit's multipage apps at: https://docs.streamlit.io/library/get-started/multipage-apps"
        )


def render_footer() -> None:
    """Renders the page footer with data attribution"""
    st.markdown("---")
    st.caption(
        "Data source: Yahoo Finance via yfinance: https://finance.yahoo.com/lookup/"
    )


# Entry point of the application
if __name__ == "__main__":
    main()
