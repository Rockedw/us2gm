Feature: creating an organisation
	In order to create opportunities and receive submissions
	As a user
	I want to create an organisation

	Background:
		Given I am a user
		# And on my homepage
		And I click 'Create Organisation'
		And I specify its name, profile-image, network and organisation type

		@javascript
		Scenario: Having entered these details
			Then I should see these details on my newly created profile
			And on my own profile I should be able to see my profile role

