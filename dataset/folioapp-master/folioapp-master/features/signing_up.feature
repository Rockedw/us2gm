Feature: signing up
	In order to use the ffolio website
	I have to sign up as a user

	Scenario: Visiting ffolio for the first time
	Given that I arrive on the ffolio site
	And I register for a ffolio account
	Then I should be on my profile
	And I should be invited to fill in my profile details
