class CreateBookmarkVisits < ActiveRecord::Migration[5.1]
  def change
    create_table :bookmark_visits do |t|
      t.integer     :user_id
      t.integer     :bookmark_id
      t.timestamps
    end
  end
end
