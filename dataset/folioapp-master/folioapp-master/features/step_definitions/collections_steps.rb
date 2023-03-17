Given(/^on my collections page$/) do
	visit "/users/#{@user.id}/collections"
end

When(/^I specify that I wish to add a new collection$/) do
	click_link 'Add A New Collection'
end

When(/^I give it a title, description and pick a cover image$/) do
	fill_in 'Title', with: "Vorticist Art"
	fill_in 'Description', with: "This is a collection of pastiches of famous vorticist poems"
	attach_file 'Image', Rails.root.join('features/images/art.jpg')
	click_button 'Add Collection'
end

Then(/^I should see that collection on my collections page$/) do
	expect(page).to have_content "Vorticist Art"
	expect(page).to have_content "This is a collection of pastiches of famous vorticist poems"
	expect(page).to have_css "img.collection-image"
end

When(/^I have clicked and edited the collection's title and description$/) do
	find('a.collection-title').trigger(:click)
	find(:css, "input").set "Hello World"
	click_button 'save'

	find('a.collection-description').trigger(:click)
	find(:css, "input").set "This is the truth"
	click_button 'save'
end

Then(/^I should see the new properties on my collections page$/) do
	page.find('h1.collection-title').should have_text 'Hello World'
	page.find('p.collection-description').should have_text 'This is the truth'
end
