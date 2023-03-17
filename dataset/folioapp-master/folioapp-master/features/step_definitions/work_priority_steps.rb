Given(/^I have a piece of work$/) do
	@user = create(:user)
	login_as @user
	@work = create(:work)
	@collection = @user.collections.first
	@user.update(work_selection: WorkSelection.create)
	@collection.works << @work
	# puts "£$£$£$£$£$£---#{@user.collections.first.works.count}----$R£$£$£"
end

Given(/^I am on the collection page for an image$/) do
	visit "/users/#{@user.id}/collections/#{@collection.id}"
end

Given(/^I specify that I wish to show it on my cover$/) do
		find(:css, "#select-work").set(true)
end

Then(/^I should see that work and its details on my profile cover\.$/) do
	visit "/users/#{@user.id}"
	# save_and_open_page
	page.find('h1.preview-title').should have_content 'HELLO WORLD'
	expect(page).to have_content 'from My Collection'
	# expect(page).to have_css 'img.selection'
end
