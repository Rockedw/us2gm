Feature: submitting to opportunities
	In order to get my work out there
	As a creative
	I want to submit to organisations' opportunities
	And send them a piece of my work

	Background:
		Given I am a user
		Given I have works
		And I am on the profile of an organisation who have published an opportunity
		And I click the 'Submit To Opportunity' link


		@javascript
		Scenario: filling in opportunity details
			When I select which work and add a message
			And I click the submit button
			And I go on my homepage
			Then I should see a thumbnail of my submission along with an acceptance status

		#Scenario: submitting the wrong kind of work

		#Trying to submit with no works

		#Scenario: getting accepted