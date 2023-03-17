Given(/^on my homepage$/) do
  visit '/'
  visit '/'
  # sleep 5
end

Given(/^I click 'Create Organisation'$/) do
	# visit current_path
	# save_and_open_page
	# click_link 'Create Organisation'
	visit '/organisations/new'
end

Given(/^I specify its name, profile\-image, network and organisation type$/) do
	# save_and_open_page
	fill_in 'Name', with: 'Notes'
	attach_file 'Image', Rails.root.join('features/images/art.jpg')
	fill_in 'Network', with: 'University of Cambridge'
	fill_in 'Description', with: 'A fortnightly magazine of essays, art and writing'
	fill_in 'Type of Organisation', with: 'Magazine'
	fill_in 'Your Role', with: 'Editor'
	click_button 'Submit'
end

Then(/^I should see these details on my newly created profile$/) do
  page.find('h1.name').should have_text 'Notes'
  page.find('p.profession').should have_text 'Magazine'
  page.find('p.network').should have_text 'Cambridge'
  page.find('p.short-bio').should have_text 'A fortnightly magazine of essays, art and writing'
  expect(page).to have_css 'img.profile-pic'
end

Then(/^on my own profile I should be able to see my profile role$/) do
	visit "/users/#{@user.id}"
	expect(page).to have_content 'Editor of Notes'
end