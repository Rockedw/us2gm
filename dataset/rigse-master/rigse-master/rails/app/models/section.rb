class Section < ApplicationRecord

  belongs_to :activity
  belongs_to :user

  has_many :offerings, :dependent => :destroy, :as => :runnable, :class_name => "Portal::Offering"

  has_many :external_activities, :as => :template

  has_one :investigation, :through => :activity

  has_many :pages, -> { order :position}, :dependent => :destroy do
    def student_only
      where('teacher_only' => false)
    end
  end

  # Generates: SELECT `page_elements`.* FROM `page_elements`
  # INNER JOIN `pages` ON `page_elements`.`page_id` = `pages`.`id`
  # WHERE `pages`.`section_id` = 2
  # ORDER BY page_elements.position ASC, page_elements.id ASC, `pages`.`position` ASC
  has_many :page_elements, :through => :pages

  acts_as_list :scope => :activity_id
  accepts_nested_attributes_for :pages, :allow_destroy => true

  acts_as_replicatable

  include Publishable
  include Changeable
  include HasEmbeddables
  include ResponseTypes

  validates_presence_of :name, :on => :create, :message => "can't be blank"

  default_value_for :name, "name of section"
  default_value_for :description, "describe the purpose of this section here..."

  send_update_events_to :investigation

  scope :like, lambda { |name|
    name = "%#{name}%"
    where("sections.name LIKE ? OR sections.description LIKE ?", name, name)
  }

  self.extend SearchableModel
  @@searchable_attributes = %w{name description}

  class <<self
    def searchable_attributes
      @@searchable_attributes
    end

    def search_list(options)
      name = options[:name]
      if (options[:include_drafts])
        sections = Section.like(name)
      else
        sections = Section.published.like(name)
      end
      portal_clazz = options[:portal_clazz] || (options[:portal_clazz_id] && options[:portal_clazz_id].to_i > 0) ? Portal::Clazz.find(options[:portal_clazz_id].to_i) : nil
      if portal_clazz
        sections = sections - portal_clazz.offerings.map { |o| o.runnable }
      end
      if options[:paginate]
        sections = sections.paginate(:page => options[:page] || 1, :per_page => options[:per_page] || 20)
      else
        sections
      end
    end
  end


  def parent
    return activity
  end

  def children
    return pages
  end

  include TreeNode

  # TODO: we have to make this container nuetral,
  # using parent / tree structure (children)
  def reportable_elements
    return @reportable_elements if @reportable_elements
    @reportable_elements = []
    unless teacher_only?
      @reportable_elements = pages.collect{|s| s.reportable_elements }.flatten
      @reportable_elements.each{|elem| elem[:section] = self}
    end
    return @reportable_elements
  end
end

#  Recent schema definition:
# create_table "sections", :force => true do |t|
#   t.datetime "created_at"
#   t.datetime "updated_at"
#   t.string   "name"
#   t.string   "description"
#   t.integer  "user_id"
#   t.integer  "position"
#   t.integer  "activity_id"
#   t.string   "uuid",             :limit => 36
# end
