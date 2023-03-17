Feature: creating opportunities
	In order to attract talented creatives
	As an organisation
	I want to create a submissions opportunity
	And specify a title, image, description and deadline

	Background:
		Given I am a user
		And I am a member of an organisation
		And I am on my organisation profile
		# And I click the 'Opportunities' link of my profile
		And I specify that I wish to create an opportunity

	@javascript
	Scenario: filling in opportunity details
		When I give my opportunity a title, image, description and deadline, and requirements
		# And I click submit
		Then I should see my opportunity on my organisation page

