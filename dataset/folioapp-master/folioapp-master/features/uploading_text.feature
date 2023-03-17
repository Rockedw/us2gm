Feature: uploading images
	In order to have a nice portfolio
	As a writer
	I want to upload text to a collection on my profile
	And specify title, thumbnail, genres and a short description


	Background:
		Given I am a user
		And on my profile
		# And I click the 'Work' link on my profile cover
		And I specify that I wish to upload text

	@javascript
	Scenario:
		When I enter text
		And I upload a thumbnail, add a title, genres and a short description
		And I click submit
		Then I should see the work's thumbnail, title, short description and genres in the default group
		And I should see the content of the work

	#Scenario: annotations/footnotes

	#Scenario: poems

	