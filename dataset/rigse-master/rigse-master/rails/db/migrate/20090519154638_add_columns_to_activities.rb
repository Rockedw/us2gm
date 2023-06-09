class AddColumnsToActivities < ActiveRecord::Migration[5.1]
  def self.up
    add_column :activities, :position, :integer
    add_column :activities, :investigation_id, :integer
  end

  def self.down
    remove_column :activities, :investigation_id
    remove_column :activities, :position
  end
end
