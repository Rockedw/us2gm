Feature: collections
	In order to have an organized portfolio
	As an artists
	I would like to add and edit collections
	And specify a title, description and a cover image

	Background:
		Given I am a user
		And on my collections page

	@javascript
	Scenario: adding a collection
		When I specify that I wish to add a new collection
		And I give it a title, description and pick a cover image
		Then I should see that collection on my collections page
		
	@javascript
	Scenario: editing a collection
		When I click the 'Toggle Edit' button
		And I have clicked and edited the collection's title and description
		Then I should see the new properties on my collections page