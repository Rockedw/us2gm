class AddIsNumericToEmbeddableDataTable < ActiveRecord::Migration[5.1]
  def self.up
    add_column :embeddable_data_tables, :is_numeric, :boolean, :default => true
  end

  def self.down
    remove_column :embeddable_data_tables, :is_numeric
  end
end
