### User Story 1: Profile Customization

**User Story:**
As a **user**, I want to **customize my profile picture and bio**, so that **I can express my personality and make my profile more engaging**.

**Acceptance Criteria:**
- Users can upload a profile picture in JPEG or PNG format, with a maximum size of 5MB.
- Users can edit and save a bio of up to 250 characters.
- Users can preview their profile before saving changes.
- If the uploaded image exceeds the size limit or is an unsupported format, the system should display an appropriate error message.
- Changes to the profile should be saved and displayed immediately after the user confirms the update.

---

### User Story 2: Password Reset

**User Story:**
As a **user**, I want to **reset my password if I forget it**, so that **I can regain access to my account securely**.

**Acceptance Criteria:**
- Users can request a password reset link by providing their registered email address.
- The system sends a password reset email containing a secure link that expires after 1 hour.
- Users can set a new password, which must meet the following criteria: at least 8 characters, one uppercase letter, one number, and one special character.
- If the link is expired or invalid, the system displays an error and prompts the user to request another link.
- After successfully resetting the password, users should be redirected to the login page with a confirmation message.

---

### User Story 3: Search Functionality

**User Story:**
As a **user**, I want to **search for content by keywords**, so that **I can quickly find the information I need**.

**Acceptance Criteria:**
- The search bar is accessible from the header of every page.
- Users can enter keywords and view a list of relevant results within 2 seconds.
- Search results are sorted by relevance, with the option to sort by date or popularity.
- If no results are found, the system displays a message suggesting the user try different keywords.
- The search functionality supports partial matches and shows suggested keywords or phrases while typing.

---
