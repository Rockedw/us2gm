Feature: Child Care

In order to efficiently manage new patient into a child care data
As a chits user


Scenario: Search Patient
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "Rizal"
And I press "Search"
Then I should see "Rizal, Jose"

Scenario: Add Patient
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "Gallo"
And I press "Search"
And I should see "NO RECORDS FOUND"
When I fill in "patient_firstname" with "Allan"
And I fill in "patient_middle" with "X"		
And I fill in "patient_lastname" with "Gallo"
And I fill in "patient_dob" with "03/04/2007"
And I select "Male" from "patient_gender"
And I fill in "patient_mother" with "maria"
And I press "Add Patient"
Then I should see "Clear Fields"


Scenario: Old Patient
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SEARCH RESULTS"
And I choose "consult_patient_id"
And I press "Select Patient"
Then I should see "VISIT DETAILS"


Scenario: Consult
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SERACH RESULTS"
And I choose "consult_patient_id" 
And I press "Select Patient"
And I should see "VISIT DETAILS"
And I check "ptgroup[]"
And I press "Save Details"
Then I should see "CHILD"

@reset_consult
Scenario:Child Care
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SERACH RESULTS"
And I choose "consult_patient_id" 
And I press "Select Patient"
And I should see "VISIT DETAILS"
And I check "ptgroup[]"
And I press "Save Details"
And I should see "CHILD"
And I click "CHILD"
Then I should see "NO REGISTRY ID FOR THIS PATIENT"

@reset_consult
Scenario:First Visit
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SEARCH RESULTS"
And I choose "consult_patient_id" 
And I press "Select Patient"
And I should see "VISIT DETAILS"
And I check "ptgroup[]"
And I press "Save Details"
And I should see "CHILD"
And I click "CHILD"
And I should see "NO REGISTRY ID FOR THIS PATIENT"
And I click "FIRST VISIT"
And I should see "CHILD CARE DATA FORM"
And I should see "Maya"
And I fill in "ccdev_date_reg" with "03/02/2010"
And I fill in "mother_px_id" with "0000001"
#And I press "Verify"
#And I should see "Patient ID does not exists" 
When I select "Housewife" from "mother_occup"
And I select "College" from "mother_educ"
And I fill in "father_name" with "Mario"
And I select "Barber" from "father_occup"
And I select "College" from "father_educ"
And I fill in "birth_weight" with "15"
And I select "Home" from "delivery_location"
When I press "Save Data"
Then I should see "FIRST VISIT DATA"

@reset_consult
Scenario:Siblings
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SEARCH RESULTS"
And I choose "consult_patient_id" 
And I press "Select Patient"
And I should see "VISIT DETAILS"
And I check "ptgroup[]"
And I press "Save Details"
And I should see "CHILD"
And I click "CHILD"
And I should see "NO REGISTRY ID FOR THIS PATIENT"
And I click "FIRST VISIT"
And I should see "CHILD CARE DATA FORM"
And I should see "Maya"
And I fill in "ccdev_date_reg" with "03/02/2010"
And I fill in "mother_px_id" with "0000001"
#And I press "Verify"
#And I should see "Patient ID does not exists" 
When I select "Housewife" from "mother_occup"
And I select "College" from "mother_educ"
And I fill in "father_name" with "Mario"
And I select "Barber" from "father_occup"
And I select "College" from "father_educ"
And I fill in "birth_weight" with "15"
And I select "Home" from "delivery_location"
When I press "Save Data"
And I should see "FIRST VISIT DATA"
And I click "SIBLINGS"
And I should see "OTHER FAMILY MEMBERS"
And I check "patients"
And I press "Add Sibling"
Then I should see "FIRST VISIT DATA"


@reset_consult
Scenario:Services
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SEARCH RESULTS"
And I choose "consult_patient_id" 
And I press "Select Patient"
And I should see "VISIT DETAILS"
And I check "ptgroup[]"
And I press "Save Details"
And I should see "CHILD"
And I click "CHILD"
And I should see "NO REGISTRY ID FOR THIS PATIENT"
And I click "FIRST VISIT"
And I should see "CHILD CARE DATA FORM"
And I should see "Maya"
And I fill in "ccdev_date_reg" with "03/02/2010"
And I fill in "mother_px_id" with "0000001"
#And I press "Verify"
#And I should see "Patient ID does not exists" 
When I select "Housewife" from "mother_occup"
And I select "College" from "mother_educ"
And I fill in "father_name" with "Mario"
And I select "Barber" from "father_occup"
And I select "College" from "father_educ"
And I fill in "birth_weight" with "15"
And I select "Home" from "delivery_location"
When I press "Save Data"
And I should see "FIRST VISIT DATA"
And I click "SIBLINGS"
And I should see "OTHER FAMILY MEMBERS"
And I check "patients"
And I press "Add Sibling"
And I should see "FIRST VISIT DATA"
And I click "SERVICES"
And I should see "services"
And I check "services[]"
And I check "vaccine[]"
When I press "Update Record"
And I should see "services[]"
And I should see "vaccine[]"
And I click "BREASTFEEDING"
Then I should see "FTITLE_CCDEV_BREASTFEED"


@reset_consult
Scenario:Breastfeeding
Given I am logged in as "user" with password "user"
And I click "TODAY'S PATIENTS"
And I should see "CONSULTS TODAY"
When I fill in "last" with "ledesma"
And I press "Search"
And I should see "SEARCH RESULTS"
And I choose "consult_patient_id" 
And I press "Select Patient"
And I should see "VISIT DETAILS"
And I check "ptgroup[]"
And I press "Save Details"
And I should see "CHILD"
And I click "CHILD"
And I should see "NO REGISTRY ID FOR THIS PATIENT"
And I click "FIRST VISIT"
And I should see "CHILD CARE DATA FORM"
And I should see "Maya"
And I fill in "ccdev_date_reg" with "03/02/2010"
And I fill in "mother_px_id" with "0000001"
When I select "Housewife" from "mother_occup"
And I select "College" from "mother_educ"
And I fill in "father_name" with "Mario"
And I select "Barber" from "father_occup"
And I select "College" from "father_educ"
And I fill in "birth_weight" with "15"
And I select "Home" from "delivery_location"
When I press "Save Data"
And I should see "FIRST VISIT DATA"
And I click "SIBLINGS"
And I should see "OTHER FAMILY MEMBERS"
And I check "patients"
And I press "Add Sibling"
And I should see "FIRST VISIT DATA"
And I click "SERVICES"
And I should see "services"
And I check "services[]"
And I check "vaccine[]"
When I press "Update Record"
And I should see "services[]"
And I should see "vaccine[]"
And I click "BREASTFEEDING"
And I should see "FTITLE_CCDEV_BREASTFEED"
And I check "bfed_month[]"
And I fill in "date_bfed_six" with "03/02/2010"
And I press "Save Breastfeeding Status"
And I should see "bfed_month"
And I click "CONSULT"
And I should see "form_consult"
And I press "End Consult"
And I should see "comfirm_end"
When I press "Yes"
Then I should see "CONSULTS TODAY"





