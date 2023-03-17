Feature: profile creation
	In order to have a full profile
	As a first-time user
	I want to fill in my details

	Background: I am a first time user
		Given I have just signed up
		And I have filled in my name, type of creative, network, and attached a profile picture


	@javascript
	Scenario: Having entered details
		Then I should see them on my newly created profile

	@javascript
	Scenario: I have finished editing
		When I click the 'Toggle Edit' button
		Then I should not be able to edit my profile information

	@javascript
	Scenario: My profile is only set to already-editable on first login
		When I log out
		And log back in
		And I visit my profile
		Then my profile should be uneditable by default


