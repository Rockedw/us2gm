Feature: so I can show the world my best work
	As an artist
	I want to have my three preferred works on my profile cover

	@javascript
	Scenario: giving an image priority
		Given I have a piece of work
		And I am on the collection page for an image
		And I specify that I wish to show it on my cover
		Then I should see that work and its details on my profile cover.