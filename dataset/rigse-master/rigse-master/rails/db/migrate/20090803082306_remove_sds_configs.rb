class RemoveSdsConfigs < ActiveRecord::Migration[5.1]
  def self.up
    drop_table :portal_sds_configs
  end

  def self.down
    create_table :portal_sds_configs do |t|
      t.integer :configurable_id
      t.string  :configurable_type
      
      t.integer :sds_id

      t.timestamps
    end
  end
end
